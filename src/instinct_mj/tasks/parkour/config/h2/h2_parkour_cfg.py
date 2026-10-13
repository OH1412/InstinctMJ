"""H2 Parkour environment, reusing the native mjlab hiking scene."""

from __future__ import annotations

import copy
import math

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationTermCfg, RewardTermCfg, SceneEntityCfg
from mjlab.sensor import GridPatternCfg, ObjRef

import instinct_mj.envs.mdp as envs_mdp
import instinct_mj.tasks.parkour.mdp as parkour_mdp
from instinct_mj.assets.unitree_h2 import (
    H2_BODY_ACTION_SCALE,
    H2_BODY_JOINT_NAMES,
    H2_CAMERA_JOINT_NAMES,
    H2_HEAD_PITCH_RANGE,
    H2_HEAD_YAW_RANGE,
    H2_JOINT_VELOCITY_LIMITS,
    get_h2_robot_cfg,
)
from instinct_mj.sensors import GroupedRayCasterCfg
from instinct_mj.tasks.parkour.config.g1.g1_parkour_target_amp_cfg import (
    instinct_g1_parkour_amp_final_cfg,
)
from instinct_mj.terrains.terrain_generator_cfg import FlatPatchSamplingCfg
from instinct_mj.terrains.trimesh import MeshCorridorTerrainCfg


def instinct_h2_parkour_env_cfg(*, play: bool = False):
    """Build a plain-PPO H2 Parkour config without G1-only AMP state."""
    cfg = instinct_g1_parkour_amp_final_cfg(play=play, shoe=False)
    cfg.scene.entities["robot"] = get_h2_robot_cfg()
    cfg.scene.entities["robot"].init_state.pos = (0.0, 0.0, 1.03)

    # Keep the policy layout compatible with HMV: 29 body actions followed by
    # an independent [head yaw, head pitch] gimbal action.
    body_action = cfg.actions["joint_pos"]
    body_action.actuator_names = H2_BODY_JOINT_NAMES
    body_action.preserve_order = True
    body_action.scale = copy.deepcopy(H2_BODY_ACTION_SCALE)
    cfg.actions["camera_yaw_pitch"] = JointPositionActionCfg(
        entity_name="robot",
        actuator_names=H2_CAMERA_JOINT_NAMES,
        preserve_order=True,
        scale=1.0,
        use_default_offset=True,
        clip={
            "head_yaw_joint": H2_HEAD_YAW_RANGE,
            "head_pitch_joint": H2_HEAD_PITCH_RANGE,
        },
    )

    body_cfg = SceneEntityCfg(name="robot", joint_names=H2_BODY_JOINT_NAMES, preserve_order=True)
    camera_cfg = SceneEntityCfg(name="robot", joint_names=H2_CAMERA_JOINT_NAMES, preserve_order=True)
    for group_name in ("policy", "critic"):
        terms = cfg.observations[group_name].terms
        terms["joint_pos"].params["asset_cfg"] = copy.deepcopy(body_cfg)
        terms["joint_vel"].params["asset_cfg"] = copy.deepcopy(body_cfg)
        # Keep all proprioceptive terms contiguous and place the depth tensor
        # last, matching the layout expected by the parkour ONNX adapter.
        depth_image = terms.pop("depth_image")
        terms["camera_yaw_pitch"] = ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            params={"asset_cfg": copy.deepcopy(camera_cfg)},
            history_length=8,
            flatten_history_dim=True,
        )
        terms["depth_image"] = depth_image

    # Attach the depth camera to the final gimbal link so both physical head
    # joints affect the rays.  The offset comes from H2's fixed camera joint.
    camera = next(sensor for sensor in cfg.scene.sensors if sensor.name == "camera")
    camera.frame = ObjRef(type="body", name="head_yaw_link", entity="robot")
    camera.ray_alignment = "base"
    camera.offset.pos = (0.08667, 0.03, 0.0099)
    camera.offset.rot = (1.0, 0.0, 0.0, 0.0)
    camera.offset.convention = "world"
    camera.noise_pipeline["depth_normalization"].depth_range = (0.5, 2.5)

    # HMV's gaze rewards use a terrain-only depth image plus a forward height
    # grid.  Keep these out of policy observations; they only shape training.
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

    target_view_params = {
        "scanner_name": "height_scanner",
        "terrain_camera_name": "camera_terrain",
        "depth_diff_threshold": 0.01,
        "confirm_steps": 1,
        "rough_threshold": 0.05,
        "rough_upper_threshold": 0.5,
    }
    dynamic_yaw_params = {
        "command_name": "base_velocity",
        "head_yaw_threshold": math.radians(15.0),
        "wz_scale": 2.0,
        "wz_max": 0.8,
        **target_view_params,
    }
    rewards = cfg.rewards["rewards"]
    rewards["dof_vel_limits"].params["velocity_limits"] = copy.deepcopy(H2_JOINT_VELOCITY_LIMITS)
    rewards["track_ang_vel_z_exp"] = RewardTermCfg(
        func=parkour_mdp.track_ang_vel_z_dynamic_exp,
        weight=1.0,
        params={"std": 0.5, **dynamic_yaw_params},
    )
    rewards["delta_yaw_reward"] = RewardTermCfg(
        func=parkour_mdp.delta_yaw_reward,
        weight=0.25,
        params=copy.deepcopy(dynamic_yaw_params),
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
            **target_view_params,
        },
    )
    rewards["camera_yaw_side_start_penalty"] = RewardTermCfg(
        func=parkour_mdp.camera_yaw_side_start_penalty,
        weight=-10.0,
        params={"side_yaw_start_abs": math.radians(15.0), **target_view_params},
    )
    rewards["head_body_yaw_l2"] = RewardTermCfg(
        func=parkour_mdp.head_body_yaw_l2,
        weight=-0.1,
        params={"forward_penalty_scale": 10.0, **target_view_params},
    )
    rewards["flat_forward_camera_pitch"] = RewardTermCfg(
        func=parkour_mdp.camera_pitch_target_reward,
        weight=1.0,
        params={"target_pitch": 0.0, "std": 0.12, "rough": False},
    )
    rewards["rough_downward_camera_pitch"] = RewardTermCfg(
        func=parkour_mdp.camera_pitch_target_reward,
        weight=1.5,
        params={"target_pitch": math.radians(20.0), "std": 0.12, "rough": True},
    )

    # Port the Monument Valley corridor mix from HMV commit 555e24d.  The
    # corridor implementation uses native MuJoCo box geoms for stable contact.
    terrain_gen = cfg.scene.terrain.terrain_generator
    terrain_gen.size = (16.0, 16.0)
    terrain_gen.num_rows = 4
    terrain_gen.num_cols = 4
    terrain_gen.sub_terrains = {
        "perlin_rough_stand": copy.deepcopy(terrain_gen.sub_terrains["perlin_rough_stand"]),
        "pyramid_stairs": copy.deepcopy(terrain_gen.sub_terrains["pyramid_stairs"]),
        "pyramid_stairs_inv": copy.deepcopy(terrain_gen.sub_terrains["pyramid_stairs_inv"]),
        "corridor_left": MeshCorridorTerrainCfg(
            proportion=1.0,
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
            first_u_turn_side=1.0,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=(0.05, 0.10, 0.15, 0.20),
                    max_height_diff=0.05,
                )
            },
        ),
        "corridor_right": MeshCorridorTerrainCfg(
            proportion=1.0,
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
            first_u_turn_side=-1.0,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=(0.05, 0.10, 0.15, 0.20),
                    max_height_diff=0.05,
                )
            },
        ),
    }
    command_cfg = cfg.commands["base_velocity"]
    command_cfg.velocity_ranges = {
        name: command_cfg.velocity_ranges.get(
            name,
            {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
        )
        for name in terrain_gen.sub_terrains
    }

    # H2 has no matching G1 motion-reference model.  The HMV H2 update uses
    # plain PPO, so remove the optional AMP/reference path entirely.
    cfg.scene.entities.pop("motion_reference", None)
    cfg.scene.sensors = tuple(sensor for sensor in cfg.scene.sensors if sensor.name != "motion_reference")
    cfg.observations.pop("amp_policy", None)
    cfg.observations.pop("amp_reference", None)
    cfg.terminations.pop("dataset_exhausted", None)
    return cfg
