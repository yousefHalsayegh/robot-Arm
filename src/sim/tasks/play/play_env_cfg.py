# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils

# from . import mdp
import sim.tasks.play.mdp as mdp
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    DeformableObjectCfg,
    RigidObjectCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from scipy.spatial.transform import Rotation
##check how to implement this 
#from isaac_so_arm101.utils.domain_randomization import domain_randomization, randomize_object_uniform


##
# Scene definition
##

def rot_cal(x_deg, y_deg, z_deg):
    target   = Rotation.from_euler('xyz', [x_deg, y_deg, z_deg], degrees=True)
    offset   = Rotation.from_euler('xyz', [90, 90, 0], degrees=True)
    result   = target * offset.inv()
    x, y, z, w = result.as_quat()
    return (w, x, y, z)

@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    screen = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/screen",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.45, -0.12, 0.36],          
            rot=[0.707, 0, 0, 0.707],      
        ),
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.51, 0.01, 0.29),      
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 0.0),   # black until texture binds
            ),
        ),
    )


    #camera
    side = CameraCfg(
        prim_path = "{ENV_REGEX_NS}/side",
        update_period=0.1,
        height=1280,
        width=720,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=6.3562, focus_distance=28.0, horizontal_aperture=12.7, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.32268, 0.24807, 0.27), rot=(0.0, 0.0, 0.38268, 0.92388), convention="opengl"),
    )
    

    back = CameraCfg(
        prim_path = "{ENV_REGEX_NS}/back",
        update_period=0.1,
        height=640,
        width=480,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.2622, focus_distance=19.0, horizontal_aperture=12.7, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.24, -0.12, 0.38), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.1, 0.1),
            pos_y=(-0.3, -0.1),
            pos_z=(0.2, 0.35),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        #need to recheck wtf is this
        # wrist = ObsTerm(
        #     func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist"), "data_type": "rgb", "normalize": False}
        # )
        # front = ObsTerm(
        #     func=mdp.image, params={"sensor_cfg": SceneEntityCfg("front"), "data_type": "rgb", "normalize": False}
        # )
    

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()

def reach_joystick(env, joystick_cfg: SceneEntityCfg, x_range:tuple[float,float], y_range:tuple[float,float], height_threshold:float):
    pass

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    #function

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.05}, weight=1.0)

    # graping_object = RewTerm(func=mdp.graping_object,  params={"threshold": 0.05}, weight=5.0)

    # grap_and_hold_object = RewTerm(func=mdp.grap_and_hold_object, params={"threshold": 0.05}, weight=10.0 )

    reach_reward = RewTerm(func=mdp.joystick_reach_reward, params={
        "std": 0.05,
    }, weight=2.0)

    touch_reward = RewTerm(func=mdp.touch_joystick, params={
        "touch": 0.05
    }, weight=4.0)
    

@configclass
class PlayEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    rewards : RewardsCfg = RewardsCfg()


    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1
        self.episode_length_s = 15.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
