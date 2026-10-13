"""HMV-parity G1 31-DoF D455 parkour configuration."""

from __future__ import annotations

import copy
import math

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationTermCfg, RewardTermCfg, SceneEntityCfg
from mjlab.sensor import GridPatternCfg, ObjRef

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
        step_height_range=(0.15, 0.20),
        step_tread_range=(0.17, 0.25),
        component_types=["u_turn"],
        first_u_turn_side=side,
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
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (height_scanner, terrain_camera)

    target_view = {
        "scanner_name": "height_scanner",
        "terrain_camera_name": "camera_terrain",
        "depth_diff_threshold": 0.01,
        "confirm_steps": 1,
        "rough_threshold": 0.05,
        "rough_upper_threshold": 0.5,
    }
    yaw = {"yaw_joint_name": "head_yaw_joint", "yaw_joint_sign": -1.0}
    dynamic_yaw = {
        "command_name": "base_velocity",
        "head_yaw_threshold": math.radians(15.0),
        "wz_scale": 2.0,
        "wz_max": 0.8,
        **yaw,
        **target_view,
    }
    rewards = cfg.rewards["rewards"]
    rewards["dof_vel_limits"].params["velocity_limits"] = copy.deepcopy(G1_29DOF_JOINT_VELOCITY_LIMITS)
    rewards["track_ang_vel_z_exp"] = RewardTermCfg(
        func=parkour_mdp.track_ang_vel_z_dynamic_exp,
        weight=1.0,
        params={"std": 0.5, **dynamic_yaw},
    )
    rewards["delta_yaw_reward"] = RewardTermCfg(
        func=parkour_mdp.delta_yaw_reward, weight=0.25, params=copy.deepcopy(dynamic_yaw)
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
    rewards["camera_yaw_target"] = RewardTermCfg(
        func=parkour_mdp.camera_yaw_target_reward,
        weight=1.0,
        params={
            "target_yaw_abs": math.radians(45.0),
            "side_yaw_start_abs": math.radians(15.0),
            "forward_yaw_abs_threshold": math.radians(10.0),
            **yaw,
            **target_view,
        },
    )
    rewards["camera_yaw_side_start_penalty"] = RewardTermCfg(
        func=parkour_mdp.camera_yaw_side_start_penalty,
        weight=-10.0,
        params={"side_yaw_start_abs": math.radians(15.0), **yaw, **target_view},
    )
    rewards["head_body_yaw_l2"] = RewardTermCfg(
        func=parkour_mdp.head_body_yaw_l2,
        weight=-0.1,
        params={"forward_penalty_scale": 10.0, **yaw, **target_view},
    )
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
        # Keep the training profile identical to the 80k run, but avoid
        # generating its full 4x10 evaluation atlas before a small GUI preview.
        cfg.scene.terrain.terrain_generator.num_rows = 1
        cfg.scene.terrain.terrain_generator.num_cols = 4
    return cfg
