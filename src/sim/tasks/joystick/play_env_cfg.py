from dataclasses import MISSING

import isaaclab.sim as sim_utils
import torch 
import torch.nn.functional as F
import numpy as np
# from . import mdp
import sim.tasks.joystick.mdp as mdp
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    DeformableObjectCfg,
    RigidObjectCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
import isaaclab.envs.manager_based_env as _mbe
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RecorderTermCfg as RecTerm
from isaaclab.managers import ActionTermCfg as ActTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CommandTermCfg as ComTerm
from isaaclab.managers import CommandTerm, ActionTerm, RecorderTerm
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, DatasetExportMode
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_mul

# from sim.enhance.managers.recorder_manager import StreamingRecorderManager
# _mbe.RecorderManager = StreamingRecorderManager
# arcade stick default pose from joint_pos_env_cfg.py InitialStateCfg
STICK_DEFAULT_POS = [0.305, -0.058, 0.0]
STICK_DEFAULT_ROT = [0.7071068, 0.0, 0.0, -0.7071068]
 
# curriculum episode lengths — step down 0.5s per stage
STAGE_EPISODE_LENGTHS = {0: 5.0, 1: 4.5, 2: 4.0, 3:3.5, 4:3.0, 5:2.5, 6:2.0, 7:1.5, 8:1.0}
 
# curriculum thresholds
UPPER_THRESHOLD = 0.80
LOWER_THRESHOLD = 0.50
WINDOW_SIZE     = 100
 
# discrete command integer codes — must match rewards.py
CMD_NEUTRAL = 0
CMD_UP      = 2
CMD_DOWN    = 3
CMD_LEFT    = 4
CMD_RIGHT   = 5
CMD_HOME    = 1
ALL_COMMANDS = [CMD_NEUTRAL, CMD_UP, CMD_DOWN, CMD_LEFT, CMD_RIGHT]
##
# Scene definition
##

class SideCameraRecorder(RecorderTerm):

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._step_counter = 0
        self._record_every_n = 100 
        
    def record_post_step(self):
        camera = self._env.scene["side"]
        rgb = camera.data.output["rgb"].clone()
        rgb_small = F.interpolate(
            rgb.permute(0, 3, 1, 2).float(), size=(180, 320), mode="bilinear"
        ).permute(0, 2, 3, 1).to(torch.uint8)
        return "side_camera_rgb", rgb_small

class JoystickActionTerm(ActionTerm):

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._actions = torch.zeros(
            env.num_envs, 1, dtype=torch.long, device=env.device
        )

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._actions

    def process_actions(self, actions: torch.Tensor):
        self._actions = actions

    def apply_actions(self):
        robot = self._env.scene[self.cfg.asset_name]
        # self._actions is now [N, 6] float — target joint positions
        robot.set_joint_position_target(self._actions.float())
        robot.write_data_to_sim()

@configclass
class JoystickActionTermCfg(ActTerm):
    class_type: type = JoystickActionTerm   # set below
    asset_name: str = "robot"
 
 
# ── command term — samples discrete joystick commands per episode ─────────────

class JoystickCommandTerm(CommandTerm):
    """
    Samples one of 5 discrete joystick commands uniformly at episode reset.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._command = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )

    def __str__(self) -> str:
        return f"JoystickCommandTerm | envs: {self.num_envs}"
    
    def _resample_command(self, env_ids: torch.Tensor):
        """Sample a new random command for the given env indices."""
        sampled = torch.tensor(
            np.random.choice(ALL_COMMANDS, size=len(env_ids)),
            dtype=torch.long,
            device=self.device,
        )
        self._command[env_ids] = sampled

    def _update_command(self):
        """Called every step — no update needed for episode-level commands."""
        pass

    def _update_metrics(self):
        """Called every step to update any logged metrics — nothing to track."""
        pass

    @property
    def command(self) -> torch.Tensor:
        """Current command tensor — [num_envs] int."""
        return self._command

    # ── command_b required by CommandTerm ─────────────────────────────────────
    @property
    def command_b(self) -> torch.Tensor:
        return self._command

@configclass
class JoystickCommandTermCfg(ComTerm):
    class_type: type = JoystickCommandTerm      # set below after class definition
    resampling_time_range: tuple = (10.0, 20.0)
    debug_vis: bool = False

# ── curriculum term — minimum success rate across all 5 commands ──────────────
 
def min_command_success_rate(
    env,
    env_ids: torch.Tensor,
    command_success_buf: dict,   # passed from training script via params
) -> torch.Tensor:
    """
    Returns the minimum success rate across all 5 command types.
    Curriculum advances when this exceeds UPPER_THRESHOLD,
    regresses when below (1 - LOWER_THRESHOLD).
 
    command_success_buf is a dict {cmd_int: deque of bool} maintained
    by the training script and passed as a parameter.
    """
    rates = []
    for cmd in ALL_COMMANDS:
        history = command_success_buf.get(cmd, [])
        if len(history) < WINDOW_SIZE:
            rates.append(0.0)   # window not full — do not advance
        else:
            rates.append(sum(history) / len(history))
    return torch.tensor(min(rates), device=env.device)
 
 
def update_episode_length(env, stage: int):
    """Called by curriculum manager when stage changes."""
    new_length = STAGE_EPISODE_LENGTHS.get(stage, 5.0)
    env.cfg.episode_length_s = new_length
 
 
# ── event term — controller position randomisation ────────────────────────────


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


    #camera
    side = CameraCfg(
        prim_path = "{ENV_REGEX_NS}/side",
        update_period=0.1,
        height=720,
        width=1280,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=6.3562, focus_distance=28.0, horizontal_aperture=12.7, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.32268, 0.24807, 0.27), rot=(0.0, 0.0, 0.38268, 0.92388), convention="opengl"),
    )
    
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    joystick_cmd = JoystickCommandTermCfg(
            debug_vis=False,
        )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joystick = JoystickActionTermCfg(
            asset_name="robot",
        )

@configclass
class ObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            joint_pos = ObsTerm(
                func=mdp.joint_pos_rel,
                params={"asset_cfg": SceneEntityCfg("robot")},
            )
            enable_corruption = False
 
        policy: PolicyCfg = PolicyCfg()
 


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success  = DoneTerm(func=mdp.success_termination, time_out=False)

    #function

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    parse_success = RewTerm(
            func=mdp.sparse,
            weight=1.0,
            params={"weight": 1.0},
        )
    step_penalty = RewTerm(
            func=mdp.step_penalty,
            weight=1.0,
            params={
                "current_budget":10, 
                "weight": 1.5,
            },
        )
    axis_bonus = RewTerm(
            func=mdp.axis_bonus,
            weight=1.0,
            params={"weight": 0.5},
        )
    vision_shaping = RewTerm(
            func=mdp.vision_shaping_reward, 
            weight=1.0, 
            params={"weight_center": 0.05, "weight_approach": 0.1}
        )
    joystick_progress_shaping = RewTerm(
                func=mdp.joystick_progress_reward, 
                weight=1.0, 
                params={"weight": 0.5}
            )
    
#TODO; fix the memory problem since it is saving too much 
@configclass
class RecordCfg(RecorderManagerBaseCfg):
    """Recorder terms for the MDP — StreamingRecorderManager is substituted
    via a module-level monkey-patch in train.py, not via this cfg."""

    dataset_export_dir_path: str = "logs/recordings2"
    dataset_filename: str = "dataset"
    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_ALL

    side_camera = RecTerm(class_type=SideCameraRecorder)

@configclass
class EventCfg:

    reset_or_restore = EventTerm(func=mdp.reset_or_restore_on_failure, mode="reset")

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
    #recorders: RecordCfg = RecordCfg()
    events: EventCfg = EventCfg()


    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1
        self.episode_length_s = 10
        self.viewer.eye = (2.5, 2.5, 1.5)
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
