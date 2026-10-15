"""HMV-parity G1 31-DoF D455 parkour configuration."""

from __future__ import annotations

import copy
import math

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationTermCfg, RewardTermCfg, SceneEntityCfg
from mjlab.sensor import CameraSensorCfg, ContactMatch, ContactSensorCfg, GridPatternCfg, ObjRef

import instinct_mj.envs.mdp as envs_mdp
import instinct_mj.tasks.parkour.mdp as parkour_mdp
from instinct_mj.assets.unitree_g1 import (
    G1_31DOF_D455_ACTION_SCALE,
    G1_31DOF_D455_CAMERA_JOINT_NAMES,
    G1_31DOF_D455_CAMERA_PITCH_DEFAULT,
    G1_31DOF_D455_CAMERA_PITCH_RANGE,
    G1_31DOF_D455_CAMERA_YAW_RANGE,
    G1_31DOF_D455_CFG,
    G1_31DOF_D455_DEPTH_FOV,
    G1_31DOF_D455_DEPTH_MIN_DISTANCE,
    G1_31DOF_D455_DEPTH_NORMALIZATION_RANGE,
    G1_31DOF_D455_FLAT_FORWARD_PITCH_TARGET,
    G1_31DOF_D455_MJCF_PATH,
    G1_31DOF_D455_POLICY_JOINT_NAMES,
    G1_31DOF_D455_ROUGH_DOWNWARD_PITCH_TARGET,
    G1_31DOF_D455_SYMMETRIC_JOINT_MAPPING,
    G1_31DOF_D455_SYMMETRIC_JOINT_REVERSE,
    G1_29DOF_JOINT_VELOCITY_LIMITS,
)
from instinct_mj.sensors import GroupedRayCasterCfg
from instinct_mj.tasks.parkour.config.g1.g1_parkour_target_amp_cfg import instinct_g1_parkour_amp_final_cfg
from instinct_mj.terrains.terrain_generator_cfg import FlatPatchSamplingCfg
from instinct_mj.terrains.trimesh import MeshCorridorTerrainCfg


def _corridor(side: float) -> MeshCorridorTerrainCfg:
    return MeshCorridorTerrainCfg(
        proportion=0.5,
        corridor_width=1.25,
        wall_height=1.5,
        perimeter_wall_height=8.0,
        wall_thickness=0.2,
        stair_guardrail_height=1.0,
        stair_guardrail_thickness=0.08,
        wall_prob=[1.0, 1.0, 1.0, 1.0],
        platform_length=1.5,
        u_turn_platform_length=1.3,
        segment_count=2,
        rise_per_stair=2.0,
        num_steps=10,
        step_height_range=(0.05, 0.20),
        step_tread_range=(0.17, 0.25),
        component_types=["u_turn"],
        first_u_turn_side=side,
        initial_spawn_backoff=1.3,
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=50,
                patch_radius=(0.05, 0.10, 0.15, 0.20),
                max_height_diff=0.05,
            )
        },
    )


def instinct_g1_d455_parkour_amp_cfg(*, play: bool = False):
    """Build G1 parkour with HMV's 29 body + 2 D455 gimbal DoFs."""
    cfg = instinct_g1_parkour_amp_final_cfg(play=play, shoe=False)
    # Capacity only: the three height-field families produce 249 real startup
    # contacts and 993 constraints. These limits do not alter contact geometry.
    cfg.sim.nconmax = 512 if play else 256
    cfg.sim.njmax = 2048 if play else 1100
    cfg.scene.entities["robot"] = copy.deepcopy(G1_31DOF_D455_CFG)

    body_action = cfg.actions["joint_pos"]
    body_action.actuator_names = G1_31DOF_D455_POLICY_JOINT_NAMES
    body_action.preserve_order = True
    body_action.scale = copy.deepcopy(G1_31DOF_D455_ACTION_SCALE)
    cfg.actions["camera_yaw_pitch"] = JointPositionActionCfg(
        entity_name="robot",
        actuator_names=G1_31DOF_D455_CAMERA_JOINT_NAMES,
        preserve_order=True,
        scale=1.0,
        use_default_offset=True,
        clip={
            "head_yaw_joint": G1_31DOF_D455_CAMERA_YAW_RANGE,
            "head_pitch_joint": G1_31DOF_D455_CAMERA_PITCH_RANGE,
        },
    )

    body_cfg = SceneEntityCfg(
        name="robot", joint_names=G1_31DOF_D455_POLICY_JOINT_NAMES, preserve_order=True
    )
    camera_cfg = SceneEntityCfg(
        name="robot", joint_names=G1_31DOF_D455_CAMERA_JOINT_NAMES, preserve_order=True
    )
    motion_body_cfg = SceneEntityCfg(
        name="motion_reference", joint_names=G1_31DOF_D455_POLICY_JOINT_NAMES, preserve_order=True
    )
    for group_name in ("policy", "critic"):
        terms = cfg.observations[group_name].terms
        terms["joint_pos"].params["asset_cfg"] = copy.deepcopy(body_cfg)
        terms["joint_vel"].params["asset_cfg"] = copy.deepcopy(body_cfg)
        depth_image = terms.pop("depth_image")
        terms["camera_yaw_pitch"] = ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            params={"asset_cfg": copy.deepcopy(camera_cfg)},
            history_length=8,
            flatten_history_dim=True,
        )
        terms["depth_image"] = depth_image
    for term_name in ("joint_pos_rel", "joint_vel"):
        cfg.observations["amp_policy"].terms[term_name].params["asset_cfg"] = copy.deepcopy(body_cfg)
        cfg.observations["amp_reference"].terms[term_name].params["asset_cfg"] = copy.deepcopy(
            motion_body_cfg
        )

    motion_reference = next(sensor for sensor in cfg.scene.sensors if sensor.name == "motion_reference")
    motion_reference.robot_model_path = G1_31DOF_D455_MJCF_PATH
    motion_reference.symmetric_augmentation_joint_mapping = G1_31DOF_D455_SYMMETRIC_JOINT_MAPPING
    motion_reference.symmetric_augmentation_joint_reverse_buf = G1_31DOF_D455_SYMMETRIC_JOINT_REVERSE

    camera = next(sensor for sensor in cfg.scene.sensors if sensor.name == "camera")
    camera.frame = ObjRef(type="body", name="d455_link", entity="robot")
    camera.ray_alignment = "base"
    camera.pattern.fovy = G1_31DOF_D455_DEPTH_FOV[1]
    camera.focal_length = 1.0
    camera.horizontal_aperture = 2.0 * math.tan(math.radians(G1_31DOF_D455_DEPTH_FOV[0]) / 2.0)
    camera.vertical_aperture = 2.0 * math.tan(math.radians(G1_31DOF_D455_DEPTH_FOV[1]) / 2.0)
    camera.min_distance = G1_31DOF_D455_DEPTH_MIN_DISTANCE
    camera.offset.pos = (0.0, 0.0, 0.0)
    camera.offset.rot = (0.9335804264972017, 0.0, 0.35836794954530027, 0.0)
    camera.offset.convention = "world"
    camera.noise_pipeline["depth_normalization"].depth_range = G1_31DOF_D455_DEPTH_NORMALIZATION_RANGE
    camera.mesh_prim_paths = ["/World/ground/", "/World/envs/env_\\d+/Robot/(?!d455_link).*"]

    height_scanner = GroupedRayCasterCfg(
        name="height_scanner",
        frame=ObjRef(type="body", name="torso_link", entity="robot"),
        pattern=GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        ray_alignment="yaw",
        max_distance=10.0,
        mesh_prim_paths=["/World/ground/"],
        debug_vis=False,
    )
    terrain_camera = copy.deepcopy(camera)
    terrain_camera.name = "camera_terrain"
    terrain_camera.mesh_prim_paths = ["/World/ground/"]
    terrain_camera.noise_pipeline = {}
    terrain_camera.data_histories = {}
    arm_contact_sensor = ContactSensorCfg(
        name="arm_contact_forces",
        primary=ContactMatch(
            mode="body",
            pattern=".*(shoulder|elbow|wrist|rubber_hand).*",
            entity="robot",
        ),
        fields=("force",),
        reduce="netforce",
        track_air_time=False,
        history_length=3,
    )
    extra_sensors = [height_scanner, terrain_camera, arm_contact_sensor]
    if play:
        # Add a render-only MuJoCo camera so mjlab's Viser camera panel can
        # provide its native depth preview and Frustum checkbox.  Training and
        # policy observations continue to use the grouped ray-cast camera.
        camera.debug_vis = False
        terrain_camera.debug_vis = False
        extra_sensors.append(
            CameraSensorCfg(
                name="d455_visualization",
                parent_body="robot/d455_link",
                # Move the render-only camera 4 cm along the calibrated
                # optical axis so the D455 housing does not occlude its image.
                # The policy ray camera remains at the original sensor origin.
                pos=(0.02972579, 0.0, -0.02676522),
                # MuJoCo cameras look along local -Z with +Y up.  This pose
                # matches the ray camera's +X-forward frame after its 42 deg
                # optical-axis correction.
                quat=(0.64597419, 0.28760624, -0.28760624, -0.64597419),
                fovy=G1_31DOF_D455_DEPTH_FOV[1],
                width=55,
                height=32,
                data_types=("depth",),
                use_textures=False,
                use_shadows=False,
                enabled_geom_groups=(0, 1, 2, 3),
                viewer_frustum_scale=G1_31DOF_D455_DEPTH_NORMALIZATION_RANGE[1],
                viewer_frustum_show_all=True,
            )
        )
    cfg.scene.sensors = tuple(cfg.scene.sensors) + tuple(extra_sensors)

    yaw = {"yaw_joint_name": "head_yaw_joint", "yaw_joint_sign": -1.0}
    live_target_yaw = {
        "command_name": "base_velocity",
        "max_head_yaw": G1_31DOF_D455_CAMERA_YAW_RANGE[1],
        **yaw,
    }
    rewards = cfg.rewards["rewards"]
    # Do not reward merely remaining alive.  Locomotion and task progress
    # should account for the positive return instead.
    rewards.pop("is_alive", None)
    # Penalize every contacting robot link except the two sole links.  The
    # contact sensor already excludes left/right ankle_roll_link, whose
    # collision capsules form the foot soles; keep a small 1 N noise floor.
    rewards["undesired_contacts"].weight = -5.0
    # Arm contacts are already included above; apply an additional -5 so a
    # shoulder/elbow/wrist/hand contact has a combined coefficient of -10.
    rewards["arm_contacts"] = RewardTermCfg(
        func=parkour_mdp.undesired_contacts,
        weight=-5.0,
        params={"sensor_name": "arm_contact_forces", "threshold": 1.0},
    )
    rewards["dont_wait"].weight = -1.0
    rewards["delayed_stuck"] = RewardTermCfg(
        func=parkour_mdp.delayed_stuck_penalty,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "window_s": 1.0,
            "min_progress": 0.04,
            "command_threshold": 0.2,
        },
    )
    # Match HMV: body yaw is guided by angular-velocity tracking rather than
    # an additional absolute heading-error penalty.  This lets the camera look
    # toward the next waypoint without over-constraining the torso heading.
    rewards.pop("heading_error", None)
    rewards["dof_vel_limits"].params["velocity_limits"] = copy.deepcopy(G1_29DOF_JOINT_VELOCITY_LIMITS)
    rewards["track_ang_vel_z_exp"] = RewardTermCfg(
        func=parkour_mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    rewards["delta_yaw_reward"] = RewardTermCfg(
        func=parkour_mdp.delta_yaw_command_reward,
        weight=0.25,
        params={"command_name": "base_velocity"},
    )
    rewards["camera_yaw_action_acc_l2"] = RewardTermCfg(
        func=parkour_mdp.camera_axis_action_acc_l2,
        weight=-0.025,
        params={"axis_id": 0, "camera_action_dim": 2},
    )
    rewards["camera_pitch_action_acc_l2"] = RewardTermCfg(
        func=parkour_mdp.camera_axis_action_acc_l2,
        weight=-0.025,
        params={"axis_id": 1, "camera_action_dim": 2},
    )
    # No directional camera-yaw objective: the head may choose its yaw freely.
    rewards.pop("camera_yaw_target", None)
    rewards.pop("camera_yaw_target_error_l2", None)
    # Corridor waypoints encode the next platform height.  This term is gated
    # to ordered-waypoint (corridor) environments and outweighs the planar
    # velocity-tracking reward (3.0 versus 2.0).
    rewards["corridor_target_height"] = RewardTermCfg(
        func=parkour_mdp.corridor_target_height_exp,
        weight=3.0,
        params={
            "command_name": "base_velocity",
            "std": 1.0,
            "foot_height_offset": 0.035,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
            ),
        },
    )
    rewards.pop("camera_yaw_side_start_penalty", None)
    rewards.pop("head_body_yaw_l2", None)
    rewards["flat_forward_camera_pitch"] = RewardTermCfg(
        func=parkour_mdp.camera_pitch_target_reward,
        weight=1.0,
        params={"target_pitch": G1_31DOF_D455_FLAT_FORWARD_PITCH_TARGET, "std": 0.12, "rough": False},
    )
    rewards["rough_downward_camera_pitch"] = RewardTermCfg(
        func=parkour_mdp.camera_pitch_target_reward,
        weight=1.5,
        params={"target_pitch": G1_31DOF_D455_ROUGH_DOWNWARD_PITCH_TARGET, "std": 0.12, "rough": True},
    )

    # HMV excludes the independently controlled camera joints from body
    # regularization.  Stage one keeps the original foundation curriculum;
    # turn corridors are introduced only in a later fine-tuning stage.
    for name in ("dof_acc_l2", "dof_vel_l2", "dof_pos_limits", "dof_vel_limits", "torque_limits"):
        rewards[name].params["asset_cfg"] = copy.deepcopy(body_cfg)
    terrain_gen = cfg.scene.terrain.terrain_generator
    terrain_gen.size = (16.0, 16.0)
    terrain_gen.num_rows = 1 if play else 4
    terrain_gen.num_cols = 2 if play else 4
    base_terrains = terrain_gen.sub_terrains
    terrain_gen.sub_terrains = {
        "perlin_rough_stand": copy.deepcopy(base_terrains["perlin_rough_stand"]),
        "pyramid_stairs": copy.deepcopy(base_terrains["pyramid_stairs"]),
        "pyramid_stairs_inv": copy.deepcopy(base_terrains["pyramid_stairs_inv"]),
    }
    terrain_gen.sub_terrains["perlin_rough_stand"].proportion = 0.20
    terrain_gen.sub_terrains["pyramid_stairs"].proportion = 0.40
    terrain_gen.sub_terrains["pyramid_stairs_inv"].proportion = 0.40
    cfg.commands["base_velocity"].random_velocity_terrain = ["perlin_rough_stand"]
    cfg.commands["base_velocity"].velocity_ranges = {
        name: {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)}
        for name in terrain_gen.sub_terrains
    }

    # Keep reset noise on the body while starting the gimbal from HMV's
    # calibrated forward pose through the entity default state.
    cfg.scene.entities["robot"].init_state.joint_pos["head_pitch_joint"] = (
        G1_31DOF_D455_CAMERA_PITCH_DEFAULT
    )
    return cfg


def instinct_g1_d455_parkour_amp_legacy_terrain_cfg(*, play: bool = False):
    """D455 task using the unchanged terrain/command setup of the 29-DoF run.

    This is a separate training profile.  The regular D455 task keeps its
    staged terrain and corridor setup intact.
    """
    cfg = instinct_g1_d455_parkour_amp_cfg(play=play)
    legacy_cfg = instinct_g1_parkour_amp_final_cfg(play=play, shoe=True)
    cfg.scene.terrain = copy.deepcopy(legacy_cfg.scene.terrain)
    cfg.commands["base_velocity"] = copy.deepcopy(legacy_cfg.commands["base_velocity"])
    if play:
        # Show the full four-level curriculum atlas during native playback.
        # Curriculum mode generates one column per terrain type, so keeping
        # both dimensions explicit exposes every terrain at every difficulty.
        cfg.scene.terrain.terrain_generator.num_rows = 4
        cfg.scene.terrain.terrain_generator.num_cols = len(
            cfg.scene.terrain.terrain_generator.sub_terrains
        )
        cfg.scene.terrain.deterministic_atlas_origins = True
        # Keep every playback robot on its assigned atlas patch after reset.
        cfg.curriculum = {}
    return cfg


def instinct_g1_d455_parkour_amp_legacy_hmv_terrain_cfg(*, play: bool = False):
    """Train on pyramid stairs (40%) and HMV U-turn corridors (60%)."""
    cfg = instinct_g1_d455_parkour_amp_legacy_terrain_cfg(play=play)
    terrain_gen = cfg.scene.terrain.terrain_generator

    # Split each terrain family equally between its two directions:
    # 20% ascending pyramid + 20% descending pyramid, and
    # 30% left corridor + 30% right corridor.
    pyramid_stairs = copy.deepcopy(terrain_gen.sub_terrains["pyramid_stairs"])
    pyramid_stairs.proportion = 0.2
    pyramid_stairs_inv = copy.deepcopy(terrain_gen.sub_terrains["pyramid_stairs_inv"])
    pyramid_stairs_inv.proportion = 0.2
    terrain_gen.sub_terrains = {
        "pyramid_stairs": pyramid_stairs,
        "pyramid_stairs_inv": pyramid_stairs_inv,
    }
    terrain_gen.sub_terrains["corridor_left"] = _corridor(1.0)
    terrain_gen.sub_terrains["corridor_left"].proportion = 0.3
    terrain_gen.sub_terrains["corridor_right"] = _corridor(-1.0)
    terrain_gen.sub_terrains["corridor_right"].proportion = 0.3

    # Curriculum generation uses one column per terrain in insertion order.
    # Keeping the configured count aligned also makes command assignment and
    # diagnostics reflect the generated four-column atlas.
    terrain_gen.num_cols = len(terrain_gen.sub_terrains)

    command_cfg = cfg.commands["base_velocity"]
    command_cfg.velocity_ranges = {
        name: copy.deepcopy(command_cfg.velocity_ranges[name])
        for name in ("pyramid_stairs", "pyramid_stairs_inv")
    }
    command_cfg.random_velocity_terrain = []
    corridor_velocity_range = {
        "lin_vel_x": (0.45, 0.8),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-1.0, 1.0),
    }
    command_cfg.velocity_ranges["corridor_left"] = copy.deepcopy(corridor_velocity_range)
    command_cfg.velocity_ranges["corridor_right"] = copy.deepcopy(corridor_velocity_range)
    reset_base = cfg.events["reset_base"]
    reset_base.func = parkour_mdp.reset_root_state_uniform_with_terrain_spawn
    reset_base.params.update(
        {
            "corridor_spawn_ratio": 0.5,
            "spawn_yaw_offset": 0.0,
            "validate_spawn_metadata": True,
        }
    )
    return cfg
