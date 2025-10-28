# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import math
import carb
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ActionTermCfg as ActionTerm,
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from Reach.tasks.manager_based.reach import ISAAC_ITEMS_DIR
from Reach.assets.ur_gripper import UR_GRIPPER_CFG
from . import mdp


from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
# ------------------------------------------------------------------------

# ------------------------------------------------------------------------

# Define explicit coordinate targets (waypoints)


PRE_GRASP = {"pos": [0.70,  0.10, 0.18]}   # above cylinder
GRASP     = {"pos": [0.70,  0.10, 0.11]}   # descend to grasp
LIFT      = {"pos": [0.70,  0.10, 0.22]}   # lift after grasp
PRE_PLACE = {"pos": [0.70, -0.30, 0.18]}   # above cube
PLACE     = {"pos": [0.70, -0.30, 0.11]}   # descend to place
RETREAT   = {"pos": [0.70, -0.30, 0.22]}   # up after release

# Waypoint list in the exact order we want:
MULTI_STEP_SEQ = [PRE_GRASP, GRASP, LIFT, PRE_PLACE, PLACE, RETREAT]
NUM_STEPS = len(MULTI_STEP_SEQ)





NUCLEUS_ASSET_ROOT_DIR = carb.settings.get_settings().get("/persistent/isaac/asset_root/cloud")
NVIDIA_NUCLEUS_DIR = f"{NUCLEUS_ASSET_ROOT_DIR}/NVIDIA"
ISAAC_NUCLEUS_DIR = f"{NUCLEUS_ASSET_ROOT_DIR}/Isaac"


def __post_init__(self):
    super().__post_init__()
    # physics pacing (optional but recommended to keep motion gentle)
    self.sim.dt = 0.005
    self.decimation = 4
    self.sim.render_interval = 2

    # Make the sequence & tolerance visible to mdp wrappers
    self.MULTI_STEP_SEQ = MULTI_STEP_SEQ
    self.MULTI_STEP_TOL = 0.05  # how close to be to advance to the next step
                 # or whatever you prefer


# ------------------------------------------------------------------------
#  Scene configuration
# ------------------------------------------------------------------------

@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Scene with robot, table, cube, cylinder, light, and ground."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    robot = UR_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=5000.0),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.0),
            rot=(0.70711, 0.0, 0.0, 0.70711),
        ),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    cube1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_ITEMS_DIR}/cube.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.7, -0.3, 0.1)),
    )

    cylender = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cylender",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_ITEMS_DIR}/cylender.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.7, 0.1, 0.1)),
    )

# ------------------------------------------------------------------------
#  MDP configuration
# ------------------------------------------------------------------------


@configclass
class ActionsCfg:
        # Replace DIK with a joint position action on the arm & keep gripper out (Events own gripper)
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_joint"],
        scale=0.25,            # small
        use_default_offset=False,
        debug_vis=False,
    )

    # keep the gripper as joint-position action on "finger_joint"
    #gripper_action = mdp.JointPositionActionCfg(
    #    asset_name="robot",
    #    joint_names=["finger_joint"],
    #    scale=0.2,
    #    use_default_offset=False,
    #    debug_vis=False,
    #)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        actions = ObsTerm(func=mdp.last_action)
        target_rel_pos = ObsTerm(
            func=mdp.current_target_rel_pos,
            params={"ee_asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"]), "sequence": MULTI_STEP_SEQ},
        )
        current_step_one_hot = ObsTerm(func=mdp.current_step_one_hot, params={"num_steps": NUM_STEPS})
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

def dik_step_to_current_target(env, env_ids=None, kp=1.2, v_max=0.08):
    import torch
    # where to go: current step’s target (you already compute this in rewards)
    ee = env.scene["robot"]
    ee_idx = ee.data.body_names.index("ee_link")
    ee_pos = ee.data.body_state_w[:, ee_idx, :3]

    # build current target from your sequence (same one used by sequence_manager)
    # For simplicity, reuse mdp.current_target_rel_pos output if you have it wired;
    # otherwise compute target_pos here based on env.buffers["curr_step"] & your sequence list.
    if not hasattr(env, "cfg") or not hasattr(env.cfg, "MULTI_STEP_SEQ"):
        return
    rel = mdp.current_target_rel_pos(env, env_ids,
                                     ee_asset_cfg=SceneEntityCfg("robot", body_names=["ee_link"]),
                                     sequence=env.cfg.MULTI_STEP_SEQ)
    # rel = target - ee_pos  => so target = ee_pos + rel
    target_pos = ee_pos + rel

    delta = target_pos - ee_pos
    v_cmd = torch.clamp(delta * kp, -v_max, v_max)
    env.action_manager._terms["arm_action"].controller.set_target_linear_velocity(v_cmd)


@configclass
class EventCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.75, 1.25), "velocity_range": (0.0, 0.0)},
    )

    reset_objects    = EventTerm(func=mdp.reset_objects_default,       mode="reset")
    sequence_manager = EventTerm(func=mdp.sequence_manager_default,    mode="step")
    lula_follow      = EventTerm(func=mdp.lula_step_to_current_target, mode="step")
    grasp_event      = EventTerm(func=mdp.grasp_event_default,         mode="step",
                                 params={"allowed_steps": (1,)})
    release_event = EventTerm(func=mdp.release_event_default,       mode="step",
                                 params={"allowed_steps": (4,)})

@configclass
class RewardsCfg:
    # Approach cylinder smoothly (bounded in [-1, 0])
    ee_to_cyl = RewTerm(
        func=mdp.ee_to_target_shaped,
        weight=1.0,
        params={
            "ee_asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"]),
            "target_cfg":   SceneEntityCfg("cylender"),
            "d_ref": 0.1,   # typical reach distance
            "alpha": 6.0,
        },
    )

    approach = RewTerm(
        func=mdp.distance_to_current_target,
        weight=1.0,
        params={"ee_asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"]),
                "sequence": MULTI_STEP_SEQ}
    )

    # After lift, focus on cube approach (gated in your curriculum / or always on for simplicity)
    #ee_to_cube = RewTerm(
    #    func=mdp.ee_to_target_shaped,
    #    weight=1.0,
    #    params={
    #        "ee_asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"]),
    #        "target_cfg":   SceneEntityCfg("cube1"),
    #        "d_ref": 0.40,
    #        "alpha": 6.0,
    #    },
    #)

    # Sparse-ish bonuses (small so they don’t explode PPO)
    lifted_bonus = RewTerm(
        func=mdp.object_lifted,
        weight=2.0,
        params={"target_cfg": SceneEntityCfg("cylender"), "lift_height": 0.12},
    )

    placed_bonus = RewTerm(
        func=mdp.object_placed_reward,
        weight=6.0,
        params={"target_cfg": SceneEntityCfg("cylender"), "goal_cfg": SceneEntityCfg("cube1"), "tol": 0.03},
    )

    # Gentle smoothness terms
    act_rate_pen = RewTerm(func=mdp.action_rate_l2, weight=-0.001)          # smaller → smoother actions
    joint_vel_pen = RewTerm(func=mdp.joint_vel_l2, weight=-0.0005,
                            params={"asset_cfg": SceneEntityCfg("robot")})
    
    action_rate = RewTerm(
        func=mdp.action_rate_l2,   # this function already exists in rewards.py
        weight=-0.001,             # same weight you were using for the action-rate penalty
    )

    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0005,
                    params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success  = DoneTerm(func=mdp.object_placed_success, time_out=False,
                        params={"target_cfg": SceneEntityCfg("cylender"), "goal_cfg": SceneEntityCfg("cube1"), "tol": 0.03})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.object_placed_success, time_out=False,
                       params={"target_cfg": SceneEntityCfg("cylender"), "goal_cfg": SceneEntityCfg("cube1")})

@configclass
class CurriculumCfg:
    action_rate = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500})
    joint_vel = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500})

@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    scene = ReachSceneCfg(num_envs=128, env_spacing=2.5)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.sim.render_interval = 2
        self.sim.dt = 0.005
        self.episode_length_s = 20.0
        self.viewer.eye = (3.5, 3.5, 3.5)



# ... your ReachSceneCfg / other classes above ...



@configclass
class ReachEnvCfg_PLAY(ManagerBasedRLEnvCfg):
    """Scripted pick-and-place demo using Differential IK (no RL policy needed)."""

    def __post_init__(self):
        super().__post_init__()

        # Required basics
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = 2
        self.scene = ReachSceneCfg(num_envs=1, env_spacing=2.0)
        self.episode_length_s = 40.0

        # Use the same observation group you train with (so dimensions match if you ever load a policy)
        from Reach.tasks.manager_based.reach.reach_env_cfg import ObservationsCfg as _Obs  # or just use ObservationsCfg directly if in the same file
        self.observations = _Obs()

        # Use the DIK actions for PLAY too (same as training ActionsCfg)
        Actions_PLAY = type("Actions_PLAY", (), {})()
        Actions_PLAY.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[".*_joint"],
            body_name="ee_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
                ik_params={"lambda_val": 0.08},
            ),
            scale=0.12,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
        )
        Actions_PLAY.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_joint"],
            scale=0.3,
            use_default_offset=False,
            debug_vis=False,
        )
        self.actions = Actions_PLAY

        # Rewards/terminations: keep minimal in PLAY (still required by the framework)
        Rewards_PLAY = type("Rewards_PLAY", (), {})()
        self.rewards = Rewards_PLAY

        Terms_PLAY = type("Terminations_PLAY", (), {})()
        Terms_PLAY.time_out = DoneTerm(func=mdp.time_out, time_out=True)
        self.terminations = Terms_PLAY

        # Scripted waypoints the DIK step will follow
        self.scripted_waypoints = [
            {"pos": [0.7,  0.10, 0.30], "grip": 0.0, "pause": 0.6},  # above cylinder
            {"pos": [0.7,  0.10, 0.10], "grip": 0.0, "pause": 0.6},  # descend
            {"pos": [0.7,  0.10, 0.10], "grip": 0.8, "pause": 0.8},  # close (pick)
            {"pos": [0.7,  0.10, 0.30], "grip": 0.8, "pause": 0.6},  # lift
            {"pos": [0.7, -0.30, 0.30], "grip": 0.8, "pause": 0.6},  # above cube
            {"pos": [0.7, -0.30, 0.10], "grip": 0.8, "pause": 0.6},  # lower
            {"pos": [0.7, -0.30, 0.10], "grip": 0.0, "pause": 0.8},  # open (place)
            {"pos": [0.7, -0.30, 0.30], "grip": 0.0, "pause": 0.5},  # retreat
        ]


        EventCfg_PLAY = type("EventCfg_PLAY", (), {})()
        EventCfg_PLAY.scripted_dik = EventTerm(
            func="Reach.tasks.manager_based.reach.reach_env_cfg:dik_step_to_current_target",
            mode="step",
        )
        self.events = EventCfg_PLAY


        # Event: call the DIK step function by module path **with a colon**
        EventCfg_PLAY = type("EventCfg_PLAY", (), {})()
        EventCfg_PLAY.scripted_motion = EventTerm(
            func="Reach.tasks.manager_based.reach.reach_env_cfg:diffik_motion_step",
            mode="interval",
            interval_range_s=(0.01, 0.01),
            is_global_time=True,
        )
        self.events = EventCfg_PLAY


# ---------- Module-level DIK step for PLAY (must be top-level, not inside a class) ----------
def diffik_motion_step(env, env_ids=None):
    import time, torch
    robot = env.scene["robot"]
    arm_action = env.action_manager._terms["arm_action"]

    if not hasattr(env.cfg, "_wp_index"):
        env.cfg._wp_index = 0
        env.cfg._last_time = time.time()

    wp = env.cfg.scripted_waypoints[env.cfg._wp_index]
    now = time.time()
    if now - env.cfg._last_time < wp.get("pause", 0.4):
        return

    ee_idx = robot.data.body_names.index("ee_link")
    ee_pos = robot.data.body_state_w[:, ee_idx, :3]
    target = torch.tensor(wp["pos"], device=env.device, dtype=torch.float32).unsqueeze(0)

    delta = target - ee_pos
    # --- clamp linear velocity to keep motion gentle ---
    kp = 1.2               # proportional gain → velocity command
    v_max = 0.08           # m/s cap (try 0.08–0.15 for filming)
    v_cmd = torch.clamp(delta * kp, -v_max, v_max)

    arm_action.controller.set_target_linear_velocity(v_cmd)

    # Gripper
    robot.set_joint_positions({"finger_joint": wp["grip"]})

    if torch.norm(delta) < 0.01:
        env.cfg._wp_index = (env.cfg._wp_index + 1) % len(env.cfg.scripted_waypoints)
        env.cfg._last_time = now
