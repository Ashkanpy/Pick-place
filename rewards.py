# rewards.py
# Ready-to-use Isaac Lab manager terms (observations, events, rewards, terminations)
# Works for both TRAIN and PLAY with Differential IK / joint actions.

from __future__ import annotations
import torch
from typing import List, Sequence
from isaaclab.managers import SceneEntityCfg


# ================================================================
# Internal helpers & buffers
# ================================================================
def _ensure_buffers(env):
    """Create a scratch dict on the env the first time we need it."""
    if not hasattr(env, "buffers") or not isinstance(getattr(env, "buffers"), dict):
        env.buffers = {}
    return env.buffers

def _device(env):
    # robust device getter
    return getattr(env, "device", getattr(getattr(env, "sim", None), "device", "cpu"))

def _ensure_curr_step(env):
    """Return a LongTensor [num_envs] and create env.buffers['curr_step'] if missing."""
    _ensure_buffers(env)
    if "curr_step" not in env.buffers:
        dev = _device(env)
        env.buffers["curr_step"] = torch.zeros(env.num_envs, dtype=torch.long, device=dev)
    return env.buffers["curr_step"]

def _ensure_sequence_buffers(env):
    """Allocates per-env buffers used by multi-step logic."""
    if not hasattr(env, "buffers"):
        env.buffers = {}

    buf = env.buffers
    device = env.device

    if "curr_step" not in buf:
        buf["curr_step"] = torch.zeros(env.num_envs, dtype=torch.long, device=device)

    if "just_advanced" not in buf:
        buf["just_advanced"] = torch.zeros(env.num_envs, dtype=torch.bool, device=device)

    if "prev_action" not in buf:
        # filled lazily on first call to action regularizer
        buf["prev_action"] = None

    if "sequence_done" not in buf:
        buf["sequence_done"] = torch.zeros(env.num_envs, dtype=torch.bool, device=device)


def _asset_pos(env, target_cfg):
    """
    Return (num_envs, 3) world position for a target that may be:
      - dict waypoint: {"pos": [x, y, z]}  -> broadcast to all envs
      - SceneEntityCfg: env.scene[name].data.root_state_w[:, :3]
      - str asset name: env.scene[str].data.root_state_w[:, :3]
    """
    import torch
    device = getattr(env, "device", "cpu")

    # Waypoint dict (sequence entry)
    if isinstance(target_cfg, dict) and "pos" in target_cfg:
        pos = torch.tensor(target_cfg["pos"], dtype=torch.float32, device=device)
        return pos.repeat(env.num_envs, 1)

    # Plain string name
    if isinstance(target_cfg, str):
        return env.scene[target_cfg].data.root_state_w[:, :3]

    # SceneEntityCfg (or any object with .name)
    if hasattr(target_cfg, "name"):
        return env.scene[target_cfg.name].data.root_state_w[:, :3]

    # Fallback: zeros (avoids crashes if something odd leaks through)
    return torch.zeros(env.num_envs, 3, dtype=torch.float32, device=device)


def _safe_last_action(env):
    """Return latest action (B, A) if available, else zeros."""
    B = env.num_envs
    device = env.device
    # Isaac Lab action manager caches last action per step
    if hasattr(env, "action_manager") and hasattr(env.action_manager, "last_action"):
        la = env.action_manager.last_action
        if isinstance(la, torch.Tensor):
            return la
    # fallback: keep our own previous action buffer if set
    if hasattr(env, "buffers") and env.buffers.get("prev_action", None) is not None:
        return env.buffers["prev_action"]
    # unknown dimensionality → guess from action manager term sizes
    act_dim = 0
    if hasattr(env, "action_manager"):
        for term in getattr(env.action_manager, "_terms", {}).values():
            if hasattr(term, "num_actions"):
                act_dim += int(term.num_actions)
    if act_dim <= 0:
        act_dim = 1
    return torch.zeros(B, act_dim, device=device, dtype=torch.float32)


# ================================================================
# Observation terms
# ================================================================

def _ee_pos(env, ee_asset_cfg: SceneEntityCfg):
    robot = env.scene[ee_asset_cfg.name]
    ee_name = ee_asset_cfg.body_names[0] if ee_asset_cfg.body_names else robot.data.body_names[-1]
    idx = robot.data.body_names.index(ee_name)
    return robot.data.body_state_w[:, idx, :3]

def joint_pos_rel(env, env_ids=None, asset_cfg: SceneEntityCfg | None = None):
    """Robot joint positions (relative / normalized if your actuator uses offsets)."""
    robot = env.scene[(asset_cfg.name if asset_cfg else "robot")]
    # (B, dof)
    return robot.data.joint_pos

def joint_vel_rel(env, env_ids=None, asset_cfg: SceneEntityCfg | None = None):
    """Robot joint velocities."""
    robot = env.scene[(asset_cfg.name if asset_cfg else "robot")]
    return robot.data.joint_vel

def last_action(env, env_ids=None):
    """Last applied action (B, A)."""
    return _safe_last_action(env)

def current_target_rel_pos(env, env_ids=None, ee_asset_cfg=None, sequence=None):
    from isaaclab.managers import SceneEntityCfg
    ee = _ee_pos(env, ee_asset_cfg or SceneEntityCfg("robot", body_names=["ee_link"]))
    step = _ensure_curr_step(env)  # <— ensures buffers + curr_step exist

    # clamp to valid range
    if sequence is None or len(sequence) == 0:
        return torch.zeros_like(ee)
    step = torch.clamp(step, 0, len(sequence) - 1)

    # build per-env target positions
    targ = torch.zeros_like(ee)
    for i in range(env.num_envs):
        entry = sequence[int(step[i])]
        if isinstance(entry, dict) and "pos" in entry:
            targ[i] = torch.tensor(entry["pos"], device=_device(env), dtype=torch.float32)
        elif hasattr(entry, "name"):  # SceneEntityCfg
            targ[i] = _asset_pos(env, entry)[i]
        else:
            targ[i] = ee[i]
    return targ - ee

def current_step_one_hot(env, env_ids=None, num_steps: int = 1):
    """One-hot encoding of current step. Returns (B, num_steps)."""
    _ensure_sequence_buffers(env)
    B = env.num_envs
    device = env.device
    step = env.buffers["curr_step"]
    one_hot = torch.zeros(B, num_steps, device=device)
    for i in range(num_steps):
        one_hot[:, i] = (step == i).float()
    return one_hot


# ================================================================
# Event terms (mode="step"/"reset"/"interval")
# ================================================================

def grasp_event(env, env_ids=None, ee_asset_cfg=None, target_cfg=None, threshold: float = 0.03):
    ee = _ee_pos(env, ee_asset_cfg or SceneEntityCfg("robot", body_names=["ee_link"]))
    tgt = _asset_pos(env, target_cfg or SceneEntityCfg("cylender"))
    close_mask = (torch.norm(ee - tgt, dim=1) < threshold)
    if close_mask.any():
        env.scene["robot"].set_joint_positions({"finger_joint": 0.7},
                                               env_ids=close_mask.nonzero(as_tuple=True)[0])

def grasp_event_default(env, env_ids=None, allowed_steps=(1,), **kwargs):
    step = _ensure_curr_step(env)  # <— safe
    ok = torch.isin(step, torch.tensor(list(allowed_steps), device=_device(env)))
    if not ok.any():
        return
    return grasp_event(env, env_ids=env_ids)

def release_event(env, env_ids=None, ee_asset_cfg=None, target_cfg=None, release_height: float = 0.12):
    # Here we release when EE is close to the PLACE waypoint (vertical tolerance)
    ee = _ee_pos(env, ee_asset_cfg or SceneEntityCfg("robot", body_names=["ee_link"]))
    place_z = ee[:, 2]  # EE height
    tgt = _asset_pos(env, target_cfg or SceneEntityCfg("cube1"))
    near_xy = (torch.norm(ee[:, :2] - tgt[:, :2], dim=1) < 0.05)
    near_z  = (torch.abs(place_z - tgt[:, 2]) < release_height)
    open_mask = near_xy & near_z
    if open_mask.any():
        env.scene["robot"].set_joint_positions({"finger_joint": 0.0},
                                               env_ids=open_mask.nonzero(as_tuple=True)[0])

def release_event_default(env, env_ids=None, allowed_steps=(4,), **kwargs):
    step = _ensure_curr_step(env)  # <— safe
    ok = torch.isin(step, torch.tensor(list(allowed_steps), device=_device(env)))
    if not ok.any():
        return
    return release_event(env, env_ids=env_ids)
# --- add near the top of rewards.py (with your other imports) ---

# rewards.py  (in the mdp package)

def lula_step_to_current_target(env, env_ids=None,
                                ee_frame: str = "ee_link",
                                kp: float = 1.0,
                                max_joint_step: float = 0.04,
                                **kwargs):
    """
    Compute joint targets with Lula IK to follow the current waypoint pose.
    - Reads env.cfg.MULTI_STEP_SEQ for targets (dict {"pos":[x,y,z]} or SceneEntityCfg)
    - Seeds IK with current joint state, low-pass to limit per-step joint motion
    """
    import torch
    try:
        from omni.isaac.motion_generation import LulaKinematicsSolver
    except Exception:
        # Lula not available in this runtime
        return

    # grab robot & current q
    robot = env.scene["robot"]
    q = robot.data.joint_pos  # (B, DoF)

    # Build target world positions from your existing helper
    from isaaclab.managers import SceneEntityCfg
    seq = getattr(env.cfg, "MULTI_STEP_SEQ", None)
    if not seq:
        return
    ee_cfg = SceneEntityCfg("robot", body_names=[ee_frame])
    ee_pos = _ee_pos(env, ee_cfg)
    rel = current_target_rel_pos(env, env_ids, ee_cfg, seq)
    target_pos = ee_pos + rel  # (B,3)

    # (Optional) keep current EE orientation; Lula needs a full pose
    # Here we build a simple "look-down" quaternion or reuse current EE quat if you have it.
    # For simplicity: identity orientation in world (adjust to your tool framing as needed)
    qwxyz = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)

    # Create (or cache) a per-env solver
    if not hasattr(env, "_lula"):
        env._lula = [None] * env.num_envs

    for i in range(env.num_envs):
        if env._lula[i] is None:
            # note: depending on your build, you may need robot description path / prim
            try:
                env._lula[i] = LulaKinematicsSolver(robot_prim_path=robot.prim_path,
                                                    end_effector_frame_name=ee_frame)
            except TypeError:
                # older/newer API variants — fall back to a single shared solver if needed
                env._lula[i] = LulaKinematicsSolver(robot_prim_path=robot.prim_path,
                                                    end_effector_frame_name=ee_frame)

        solver = env._lula[i]

        # Build target pose (xyz + qwxyz). Lula APIs vary; a common pattern is a dict or tuple
        xyz = target_pos[i].detach().cpu().numpy()
        q_wxyz = qwxyz[i].detach().cpu().numpy()

        # seed with current q[i]
        q0 = q[i].detach().cpu().numpy()

        # Solve IK (APIs differ slightly; keep this conceptual)
        try:
            q_sol = solver.compute_inverse_kinematics(target_position=xyz,
                                                      target_orientation=q_wxyz,
                                                      initial_guess=q0)
        except Exception:
            # if the API returns a struct, adapt accordingly
            continue

        # Low-pass / step-limit joint moves to avoid jerks
        q_sol_t = torch.tensor(q_sol, device=env.device, dtype=torch.float32)
        dq = torch.clamp(q_sol_t - q[i], -max_joint_step, max_joint_step)
        q_cmd = q[i] + kp * dq

        # Send joint targets (position control)
        robot.set_joint_positions(q_cmd.unsqueeze(0), env_ids=torch.tensor([i], device=env.device))



def dik_step_to_current_target(env, env_ids=None, kp: float = 0.8, v_max: float = 0.05, **kwargs):
    arm_term = getattr(env.action_manager, "_terms", {}).get("arm_action", None)
    if arm_term is None or not hasattr(arm_term, "controller"):
        return
    seq = getattr(env.cfg, "MULTI_STEP_SEQ", None)
    if not seq:
        return
    _ensure_curr_step(env)  # <— ensures the buffer exists

    from isaaclab.managers import SceneEntityCfg
    ee_cfg = SceneEntityCfg("robot", body_names=["ee_link"])
    ee_pos = _ee_pos(env, ee_cfg)
    rel = current_target_rel_pos(env, env_ids, ee_cfg, seq)
    target = ee_pos + rel
    delta = target - ee_pos
    v_cmd = torch.clamp(delta * float(kp), -float(v_max), float(v_max))
    arm_term.controller.set_target_linear_velocity(v_cmd)


# -----------------------------------------------
# New implementation: reset object root poses via
# Omni Isaac Core DynamicCuboid.set_world_pose()
# -----------------------------------------------
def reset_root_state(env, env_ids=None, asset_cfgs=None):
    """
    Reset rigid objects to their configured initial poses (world coordinates).

    - Reads local init poses from env.cfg.scene.<asset>.init_state
    - Adds per-env origin offsets to produce world poses
    - Uses DynamicCuboid.set_world_pose (XYZW) when available
    - Falls back to RigidObject.write_root_link_pose_to_sim ([x y z qw qx qy qz])

    Signature: func(env, env_ids=None, asset_cfgs=None)
    """
    import numpy as np
    import torch
    from isaaclab.managers import SceneEntityCfg

    device = env.device
    num_envs = getattr(env.scene.cfg, "num_envs", env.num_envs)

    # ---- env_ids as LongTensor on correct device ----
    if env_ids is None:
        env_ids = torch.arange(num_envs, dtype=torch.long, device=device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=device)
    else:
        env_ids = env_ids.to(dtype=torch.long, device=device)

    # ---- which assets to reset ----
    if asset_cfgs is None:
        asset_cfgs = [SceneEntityCfg("cube1"), SceneEntityCfg("cylender")]

    # ---- per-env world origins (default to zeros if not present) ----
    env_origins = None
    # Try common attributes
    if hasattr(env.scene, "env_origins"):
        env_origins = env.scene.env_origins
    elif hasattr(env, "env_origins"):
        env_origins = env.env_origins
    # Fallback zeros
    if env_origins is None:
        env_origins = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
    else:
        env_origins = torch.as_tensor(env_origins, dtype=torch.float32, device=device)

    # ---- helper: read init pose from env.cfg.scene (authoritative) ----
    def _cfg_init_pose_world(asset_name: str, idx: int):
        # read local init pose from the registered scene config
        try:
            scene_cfg = env.cfg.scene
            # access by attribute if exists, else dict-style indexing if you store differently
            asset_cfg = getattr(scene_cfg, asset_name)
            init = asset_cfg.init_state
            pos_local = np.array(init.pos, dtype=np.float32)               # (x, y, z) local
            # assume stored as (w, x, y, z) unless you know otherwise
            w, x, y, z = init.rot
            rot_wxyz = np.array([w, x, y, z], dtype=np.float32)
        except Exception:
            # fallback to the live asset cfg if needed
            asset = env.scene[asset_name]
            init = asset.cfg.init_state
            pos_local = np.array(getattr(init, "pos", (0.0, 0.0, 0.0)), dtype=np.float32)
            w, x, y, z = getattr(init, "rot", (1.0, 0.0, 0.0, 0.0))
            rot_wxyz = np.array([w, x, y, z], dtype=np.float32)

        # convert to world using this env's origin
        pos_world = pos_local + env_origins[idx].cpu().numpy()
        return pos_world, rot_wxyz

    # ---- try DynamicCuboid (Isaac Sim runtime) ----
    use_core = True
    try:
        from omni.isaac.core.objects import DynamicCuboid  # lazy import
    except Exception:
        use_core = False

    if use_core and not hasattr(env, "_dyn_cuboid_cache"):
        env._dyn_cuboid_cache = {}

    if use_core:
        # Isaac Sim path: set world pose with XYZW
        for idx in env_ids.tolist():
            for cfg in asset_cfgs:
                asset_name = cfg.name
                asset = env.scene[asset_name]
                prim_path = asset.prim_path  # authoritative prim path for this env
                pos_world, rot_wxyz = _cfg_init_pose_world(asset_name, idx)

                # DynamicCuboid expects XYZW
                w, x, y, z = rot_wxyz
                rot_xyzw = np.array([x, y, z, w], dtype=np.float32)

                dc = env._dyn_cuboid_cache.get(prim_path)
                if dc is None:
                    safe_name = f"dc_{prim_path.replace('/', '_')}"
                    try:
                        dc = DynamicCuboid(prim_path=prim_path, name=safe_name)
                    except TypeError:
                        dc = DynamicCuboid(prim_path=prim_path)
                    env._dyn_cuboid_cache[prim_path] = dc

                dc.set_world_pose(position=pos_world, orientation=rot_xyzw)
        return

    # ---- Isaac Lab fallback: write packed [x y z qw qx qy qz] with world positions ----
    for cfg in asset_cfgs:
        asset_name = cfg.name
        asset = env.scene[asset_name]

        # build per-env world poses
        pos_list = []
        rot_list = []
        for idx in range(num_envs):
            pos_w, rot_wxyz = _cfg_init_pose_world(asset_name, idx)
            pos_list.append(pos_w)
            rot_list.append(rot_wxyz)

        pos0 = torch.tensor(np.stack(pos_list, axis=0), dtype=torch.float32, device=device)  # [B,3]
        rot0_wxyz = torch.tensor(np.stack(rot_list, axis=0), dtype=torch.float32, device=device)  # [B,4]

        root_pose = torch.cat([pos0, rot0_wxyz], dim=1)  # [B,7] -> [x y z qw qx qy qz]
        asset.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)

        # zero velocities
        lin0 = torch.zeros_like(pos0)
        ang0 = torch.zeros_like(pos0)
        try:
            asset.write_root_velocity_to_sim(lin0, ang0, env_ids=env_ids)
        except TypeError:
            root_vel = torch.cat([lin0, ang0], dim=1)
            asset.write_root_velocity_to_sim(root_vel, env_ids=env_ids)

    if hasattr(env.scene, "write_data_to_sim"):
        env.scene.write_data_to_sim()
    env.sim.step()



    # Optional: if you want to ensure velocities are zeroed, you can also do:
    #   dc.set_linear_velocity(np.zeros(3, dtype=np.float32))
    #   dc.set_angular_velocity(np.zeros(3, dtype=np.float32))
    #
    # But since you specifically requested only set_world_pose(), we leave it at that.

def reset_objects_default(env, env_ids=None):
    """Wrapper for EventTerm: no params passed from config.
    Hard-codes the assets to reset (cube1, cylender)."""
    asset_cfgs = [SceneEntityCfg("cube1"), SceneEntityCfg("cylender")]
    return reset_root_state(env, env_ids=env_ids, asset_cfgs=asset_cfgs)


def sequence_manager(env, env_ids=None, sequence=None, tolerance: float = 0.05):
    from isaaclab.managers import SceneEntityCfg
    if sequence is None or len(sequence) == 0:
        return
    rel = current_target_rel_pos(env, env_ids, SceneEntityCfg("robot", body_names=["ee_link"]), sequence)
    arrived = (torch.norm(rel, dim=1) < tolerance)
    curr = _ensure_curr_step(env)  # <— safe
    next_step = curr + arrived.long()
    _ensure_buffers(env)["curr_step"] = torch.clamp(next_step, 0, len(sequence) - 1)

def sequence_manager_default(env, env_ids=None, **kwargs):
    seq = getattr(env.cfg, "MULTI_STEP_SEQ", None)
    if not seq:
        return
    tol = getattr(env.cfg, "MULTI_STEP_TOL", 0.05)
    return sequence_manager(env, env_ids=env_ids, sequence=seq, tolerance=tol)


# ================================================================
# Reward terms
# ================================================================

def distance_to_current_target(env, env_ids=None, ee_asset_cfg: SceneEntityCfg | None = None,
                               sequence: List[SceneEntityCfg | dict] | None = None):
    """Negative distance to current target (shaping). Returns (B,) float."""
    if not sequence:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    _ensure_sequence_buffers(env)
    ee = _ee_pos(env, ee_asset_cfg or SceneEntityCfg("robot", body_names=["ee_link"]))
    step = env.buffers["curr_step"]

    dist = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    for i, tgt in enumerate(sequence):
        mask = (step == i)
        if not mask.any():
            continue
        tgt_pos = _asset_pos(env, tgt)
        dist[mask] = torch.norm(ee[mask] - tgt_pos[mask], dim=1)
    return -dist

def step_completion_bonus(env, env_ids=None, bonus: float = 5.0):
    """Bonus when the sequence manager advanced the step in this tick."""
    _ensure_sequence_buffers(env)
    return env.buffers["just_advanced"].float() * float(bonus)

def collision_penalty(env, env_ids=None, ee_asset_cfg: SceneEntityCfg | None = None,
                      obstacles: Sequence[SceneEntityCfg] | None = None, min_clearance: float = 0.08):
    """Penalize proximity to obstacles: sum(ReLU(min_clearance - dist))."""
    if not obstacles:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    ee = _ee_pos(env, ee_asset_cfg or SceneEntityCfg("robot", body_names=["ee_link"]))
    total = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    for cfg in obstacles:
        pos = _asset_pos(env, cfg)
        d = torch.norm(ee - pos, dim=1)
        total += torch.relu(min_clearance - d)
    return -total  # negative penalty

def object_lifted(env, env_ids=None, target_cfg=None, lift_height=0.12):
    obj = _asset_pos(env, target_cfg)
    return (obj[:, 2] > lift_height).float()


def object_placed_reward(env, env_ids=None, target_cfg=None, goal_cfg=None, tol=0.03):
    obj = _asset_pos(env, target_cfg); goal = _asset_pos(env, goal_cfg)
    dist = torch.norm(obj - goal, dim=1)
    return (dist < tol).float()


def action_rate_l2(env, env_ids=None):
    if not hasattr(env, "buffers"):
        env.buffers = {}
    last = getattr(env.action_manager, "last_action", None)
    if last is None or not isinstance(last, torch.Tensor):
        return torch.zeros(env.num_envs, device=env.device)
    prev = env.buffers.get("prev_action", None)
    if prev is None or prev.shape != last.shape:
        env.buffers["prev_action"] = last.clone()
        return torch.zeros(env.num_envs, device=env.device)
    diff = last - prev
    env.buffers["prev_action"] = last.clone()
    return torch.sum(diff * diff, dim=1)  # float (B,)


# ================================================================
# Termination terms
# ================================================================


def time_out(env, env_ids=None, time_out=True):
    return (env.episode_length_buf >= env.max_episode_length)


def sequence_completed_termination(env, env_ids=None, num_steps: int | None = None):
    """Terminate when the last step has been reached (if you want finite-horizon sequences)."""
    _ensure_sequence_buffers(env)
    if num_steps is None or num_steps <= 0:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env.buffers["curr_step"] >= (num_steps - 1)


def object_placed_success(env, env_ids=None, target_cfg=None, goal_cfg=None, tol=0.03):
    obj = _asset_pos(env, target_cfg); goal = _asset_pos(env, goal_cfg)
    dist = torch.norm(obj - goal, dim=1)
    return (dist < tol)


def _tanh_shaping(dist, alpha=6.0, d_ref=0.30):
    """
    Convert a distance (m) into a bounded reward in [-1, 0], approaching 0 as dist→0.
    alpha controls slope near 0; d_ref normalizes typical workspace distances.
    """
    x = torch.clamp(dist / d_ref, 0.0, 2.0)
    return -torch.tanh(alpha * x)  # in (-1, 0]; smoother than -dist

def ee_to_target_shaped(env, env_ids=None, ee_asset_cfg=None, target_cfg=None, d_ref=0.30, alpha=6.0):
    ee = _ee_pos(env, ee_asset_cfg or SceneEntityCfg("robot", body_names=["ee_link"]))
    tgt = _asset_pos(env, target_cfg)
    dist = torch.norm(ee - tgt, dim=1)
    return _tanh_shaping(dist, alpha=alpha, d_ref=d_ref)  # float (B,)

def joint_vel_l2(env, env_ids=None, asset_cfg: SceneEntityCfg=None):
    robot = env.scene[(asset_cfg.name if asset_cfg else "robot")]
    vel = robot.data.joint_vel
    return torch.sum(vel * vel, dim=1)

