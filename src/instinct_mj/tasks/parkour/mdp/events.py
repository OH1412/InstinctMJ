from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mjlab.envs.mdp as envs_mdp
import torch
from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


def randomize_rigid_body_material(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    static_friction_range: tuple[float, float] = (0.3, 1.0),
    dynamic_friction_range: tuple[float, float] = (0.3, 1.0),
    make_consistent: bool = True,
) -> None:
    slide_friction_range = (
        min(static_friction_range[0], dynamic_friction_range[0]),
        max(static_friction_range[1], dynamic_friction_range[1]),
    )
    envs_mdp.dr.geom_friction(
        env=env,
        env_ids=env_ids,
        ranges=slide_friction_range,
        operation="abs",
        asset_cfg=asset_cfg,
        shared_random=make_consistent,
    )


def push_by_setting_velocity_without_stand(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    velocity_threshold: float = 0.15,
) -> None:
    """Push the asset by setting root velocity. No push when standing still."""
    asset: Entity = env.scene[asset_cfg.name]
    vel_w = asset.data.root_link_vel_w[env_ids]

    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    add_vel = sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=env.device)

    cmd = env.command_manager.get_command(command_name)[env_ids]
    lin_vel = torch.norm(cmd[:, :2], dim=1) > velocity_threshold
    ang_vel = torch.abs(cmd[:, 2]) > velocity_threshold

    should_push = torch.logical_or(lin_vel, ang_vel).float().unsqueeze(-1)
    vel_w += add_vel * should_push
    asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)


def reset_root_state_uniform_with_terrain_spawn(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    corridor_spawn_ratio: float = 0.5,
    spawn_height_offset: float | None = None,
    spawn_yaw_offset: float = 0.0,
    validate_spawn_metadata: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset normally, with optional safe platform spawns on corridor terrain."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    asset: Entity = env.scene[asset_cfg.name]
    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    root_states = default_root_state[env_ids].clone()

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")]
    ranges = torch.tensor(range_list, dtype=root_states.dtype, device=env.device)
    pose_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)
    positions = root_states[:, :3] + env.scene.env_origins[env_ids] + pose_samples[:, :3]
    orientations = quat_mul(
        root_states[:, 3:7],
        quat_from_euler_xyz(pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]),
    )

    terrain = env.scene["terrain"]
    generator = getattr(terrain, "terrain_generator", None)
    get_subterrain_cfg = getattr(generator, "get_subterrain_cfg", None)
    has_metadata_api = get_subterrain_cfg is not None and hasattr(terrain, "terrain_levels") and hasattr(
        terrain, "terrain_types"
    )
    if validate_spawn_metadata and not has_metadata_api:
        raise RuntimeError("Terrain generator does not expose per-subterrain spawn metadata.")

    if has_metadata_api and corridor_spawn_ratio > 0.0:
        subterrain_cfgs = get_subterrain_cfg(terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids])
        use_platform = torch.rand(len(env_ids), device=env.device) < corridor_spawn_ratio
        for buffer_index, subterrain_cfg in enumerate(subterrain_cfgs):
            if not use_platform[buffer_index].item() or subterrain_cfg is None:
                continue
            candidate_points = getattr(subterrain_cfg, "spawn_points_local", None)
            if not candidate_points:
                continue

            point_index = int(torch.randint(len(candidate_points), (1,), device=env.device).item())
            point = torch.tensor(candidate_points[point_index], dtype=positions.dtype, device=env.device)
            env_origin = env.scene.env_origins[env_ids[buffer_index]]
            root_height = root_states[buffer_index, 2] if spawn_height_offset is None else spawn_height_offset
            positions[buffer_index, :2] = env_origin[:2] + point[:2]
            positions[buffer_index, 2] = env_origin[2] + point[2] + root_height

            candidate_yaws = getattr(subterrain_cfg, "spawn_yaws", None)
            if candidate_yaws is not None and len(candidate_yaws) == len(candidate_points):
                spawn_yaw = float(candidate_yaws[point_index])
            else:
                spawn_yaw = float(getattr(subterrain_cfg, "spawn_yaw", 0.0))
            spawn_yaw += spawn_yaw_offset + float(pose_samples[buffer_index, 5])
            yaw = torch.tensor([spawn_yaw], dtype=positions.dtype, device=env.device)
            spawn_delta = quat_from_euler_xyz(
                pose_samples[buffer_index : buffer_index + 1, 3],
                pose_samples[buffer_index : buffer_index + 1, 4],
                yaw,
            )
            orientations[buffer_index] = quat_mul(
                root_states[buffer_index : buffer_index + 1, 3:7], spawn_delta
            )[0]

    velocity_ranges = torch.tensor(
        [velocity_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")],
        dtype=root_states.dtype,
        device=env.device,
    )
    velocity_samples = sample_uniform(
        velocity_ranges[:, 0], velocity_ranges[:, 1], (len(env_ids), 6), device=env.device
    )
    velocities = root_states[:, 7:13] + velocity_samples
    asset.write_root_link_pose_to_sim(torch.cat((positions, orientations), dim=-1), env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)
