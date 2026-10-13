"""Native MuJoCo asset definition for the Unitree H2 humanoid."""

from __future__ import annotations

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


_ASSET_DIR = Path(__file__).resolve().parent / "resources" / "unitree_h2"
H2_XML = _ASSET_DIR / "h2.xml"
H2_CAMERA_JOINT_NAMES = ("head_yaw_joint", "head_pitch_joint")
H2_HEAD_YAW_RANGE = (-1.7453, 1.7453)
H2_HEAD_PITCH_RANGE = (-0.5236, 0.83775)
H2_BODY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_roll_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def get_h2_spec() -> mujoco.MjSpec:
    """Load H2's MJCF; its ``assets`` directory is beside the XML file."""
    return mujoco.MjSpec.from_file(str(H2_XML))


H2_ACTUATOR_LEG = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_pitch.*", ".*_hip_roll.*", ".*_hip_yaw.*", ".*_knee.*"),
    stiffness=200.0, damping=4.0, effort_limit=360.0, armature=0.01,
)
H2_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_ankle_roll.*",), stiffness=40.0, damping=2.0, effort_limit=19.0, armature=0.01,
)
H2_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_ankle_pitch.*",), stiffness=40.0, damping=2.0, effort_limit=66.0, armature=0.01,
)
H2_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
    target_names_expr=("waist_.*",), stiffness=150.0, damping=3.0, effort_limit=120.0, armature=0.01,
)
H2_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_shoulder_pitch.*", ".*_shoulder_roll.*", ".*_shoulder_yaw.*", ".*_elbow.*", ".*_wrist_roll.*"
    ),
    stiffness=40.0, damping=2.0, effort_limit=54.0, armature=0.01,
)
H2_ACTUATOR_WRIST = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_wrist_pitch.*", ".*_wrist_yaw.*"),
    stiffness=20.0, damping=1.0, effort_limit=25.0, armature=0.01,
)
H2_ACTUATOR_HEAD = BuiltinPositionActuatorCfg(
    target_names_expr=("head_yaw_joint", "head_pitch_joint"),
    stiffness=40.0,
    damping=2.0,
    effort_limit=50.0,
    armature=0.01,
)

H2_HOME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 1.03),
    joint_pos={
        ".*_hip_pitch_joint": -0.25,
        ".*_knee_joint": 0.5,
        ".*_ankle_pitch_joint": -0.25,
        ".*_shoulder_pitch_joint": 0.35,
        ".*_elbow_joint": 0.87,
        "left_shoulder_roll_joint": 0.18,
        "right_shoulder_roll_joint": -0.18,
    },
    joint_vel={".*": 0.0},
)

H2_FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    contype=1,
    conaffinity=1,
    condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
    priority={r"^(left|right)_foot[1-7]_collision$": 1, ".*_collision": 0},
    friction={r"^(left|right)_foot[1-7]_collision$": (0.6,), ".*_collision": (1.0,)},
)

H2_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        H2_ACTUATOR_LEG,
        H2_ACTUATOR_ANKLE_ROLL,
        H2_ACTUATOR_ANKLE_PITCH,
        H2_ACTUATOR_WAIST,
        H2_ACTUATOR_ARM,
        H2_ACTUATOR_WRIST,
        H2_ACTUATOR_HEAD,
    ),
    soft_joint_pos_limit_factor=0.9,
)


def get_h2_robot_cfg() -> EntityCfg:
    """Return a fresh H2 entity configuration."""
    return EntityCfg(
        init_state=H2_HOME,
        collisions=(H2_FULL_COLLISION,),
        spec_fn=get_h2_spec,
        articulation=H2_ARTICULATION,
    )


H2_ACTION_SCALE: dict[str, float] = {}
for actuator in H2_ARTICULATION.actuators:
    assert isinstance(actuator, BuiltinPositionActuatorCfg)
    assert actuator.effort_limit is not None and actuator.stiffness is not None
    for pattern in actuator.target_names_expr:
        H2_ACTION_SCALE[pattern] = 0.25 * actuator.effort_limit / actuator.stiffness

H2_BODY_ACTION_SCALE = {
    pattern: scale for pattern, scale in H2_ACTION_SCALE.items() if not pattern.startswith("head_")
}


H2_JOINT_VELOCITY_LIMITS = {
    ".*_hip_pitch_joint": 20.0,
    ".*_hip_roll_joint": 20.0,
    ".*_hip_yaw_joint": 20.0,
    ".*_knee_joint": 20.0,
    ".*_ankle_.*_joint": 20.0,
    "waist_.*_joint": 20.0,
    ".*_shoulder_.*_joint": 20.0,
    ".*_elbow_joint": 20.0,
    ".*_wrist_.*_joint": 20.0,
    "head_.*_joint": 10.0,
}
