# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp
from isaaclab.assets import RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# from isaaclab.managers NotImplementedError
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg
from isaaclab.sim.schemas import ArticulationRootPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from sim.robots import  SO_ARM101_CFG  # noqa: F401
from sim.tasks.joystick.play_env_cfg import PlayEnvCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip



@configclass
class SoArm101LiftCubeEnvCfg(PlayEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set so arm as robot
        self.scene.robot = SO_ARM101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # override actions
        # self.actions.arm_action = mdp.JointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"],
        #     scale=0.5,
        #     use_default_offset=True,
        # )
        # self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["gripper"],
        #     open_command_expr={"gripper": 0.5},
        #     close_command_expr={"gripper": 0.0},
        # )

        # Set the arcade stick as an object
        self.scene.object = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/object",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=[0.305, -0.058, 0.0],
                rot=[0.7071068, 0.0, 0.0, -0.7071068],
                joint_pos={".*": 0.0},  # joystick starts centered
                    ),
            actuators={
                "joystick": ImplicitActuatorCfg(
                    joint_names_expr=["PivotX", "PivotY"],  
                    stiffness=3.5,  
                    damping=0.05,    
                    friction = 0.1, 
                ),
            },
            spawn=UsdFileCfg(
                usd_path="src/assets/arcade_stick_physics.usdc",
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                 articulation_props=ArticulationRootPropertiesCfg(
                    solver_position_iteration_count=64,
                    solver_velocity_iteration_count=4,
                ),  
            ),
           
        )

        #Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                #robot end-effector
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.01, 0.0, -0.09],
                    ),
                ),
                #joystik knob
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/object/joystick_Upper",
                    name="joystick_tip",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.06296],
                    ),
                ),
            ],
        )