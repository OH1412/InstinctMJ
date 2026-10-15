from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from mjlab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from mjlab.sensor import ContactSensor, RayCastSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import resolve_matching_names_values

from instinct_mj.envs.mdp.rewards.regularizations import (
    applied_torque_limits_by_ratio as _applied_torque_limits_by_ratio_general,
)
from instinct_mj.envs.mdp.rewards.regularizations import motors_power_square as _motors_power_square_general

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


def track_lin_vel_xy_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of reference linear velocity (x/y in body frame)."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    lin_vel_error = torch.sum(
        torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of reference yaw angular velocity (body frame)."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    ang_vel_error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def roughness_gate_from_scanner(
    env: ManagerBasedRlEnv,
    sensor_name: str = "height_scanner",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    x_window: tuple[float, float] = (0.15, 0.65),
    y_abs_limit: float = 0.20,
    rough_threshold: float = 0.05,
    rough_upper_threshold: float = 0.5,
) -> torch.Tensor:
    """Return a binary terrain-roughness gate from the forward ray grid."""
    sensor: RayCastSensor = env.scene[sensor_name]
    asset: Entity = env.scene[asset_cfg.name]
    hit_pos_w = sensor.data.hit_pos_w
    hit_rel_w = hit_pos_w - sensor.data.pos_w.unsqueeze(1)

    heading = asset.data.heading_w.unsqueeze(1)
    cos_heading = torch.cos(heading)
    sin_heading = torch.sin(heading)
    hit_x_b = cos_heading * hit_rel_w[..., 0] + sin_heading * hit_rel_w[..., 1]
    hit_y_b = -sin_heading * hit_rel_w[..., 0] + cos_heading * hit_rel_w[..., 1]
    valid = sensor.data.distances >= 0.0
    valid &= (hit_x_b >= x_window[0]) & (hit_x_b <= x_window[1])
    valid &= torch.abs(hit_y_b) <= y_abs_limit

    z_hits = hit_pos_w[..., 2]
    has_valid = valid.any(dim=1)
    z_max = torch.where(valid, z_hits, torch.full_like(z_hits, -torch.inf)).amax(dim=1)
    z_min = torch.where(valid, z_hits, torch.full_like(z_hits, torch.inf)).amin(dim=1)
    roughness = torch.where(has_valid, z_max - z_min, torch.zeros_like(z_max))
    return ((roughness >= rough_threshold) & (roughness < rough_upper_threshold)).float()


def _camera_depth_lr(
    env: ManagerBasedRlEnv,
    sensor_name: str = "camera_terrain",
    side_fraction: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    camera = env.scene[sensor_name]
    depth = camera.data.output["distance_to_image_plane"][..., 0]
    width = depth.shape[2]
    side_width = max(1, int(round(width * side_fraction)))
    left_depth = depth[:, :, :side_width].mean(dim=(1, 2))
    right_depth = depth[:, :, width - side_width :].mean(dim=(1, 2))
    return left_depth, right_depth, right_depth - left_depth


def _update_depth_target_view(
    env: ManagerBasedRlEnv,
    scanner_name: str = "height_scanner",
    terrain_camera_name: str = "camera_terrain",
    depth_diff_threshold: float = 0.01,
    confirm_steps: int = 1,
    rough_threshold: float = 0.05,
    rough_upper_threshold: float = 0.5,
) -> torch.Tensor:
    """Select forward/left/right view from terrain depth, once per environment step.

    The returned convention follows the depth image: ``-1`` is left and ``+1``
    is right.  H2's positive yaw joint rotates left, so reward functions convert
    this view convention to joint coordinates explicitly.
    """
    step = int(env.common_step_counter)
    if getattr(env, "_h2_yaw_target_last_step", None) == step:
        return env._h2_yaw_target_view

    if not hasattr(env, "_h2_yaw_target_view"):
        env._h2_yaw_target_view = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
        env._h2_yaw_pending_view = torch.zeros_like(env._h2_yaw_target_view)
        env._h2_yaw_pending_count = torch.zeros_like(env._h2_yaw_target_view)
        env._h2_yaw_prev_front_flat = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._h2_yaw_flat_ready = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    target = env._h2_yaw_target_view
    pending = env._h2_yaw_pending_view
    pending_count = env._h2_yaw_pending_count
    reset_mask = env.episode_length_buf <= 1
    target[reset_mask] = 0
    pending[reset_mask] = 0
    pending_count[reset_mask] = 0

    gate = roughness_gate_from_scanner(
        env,
        sensor_name=scanner_name,
        rough_threshold=rough_threshold,
        rough_upper_threshold=rough_upper_threshold,
    )
    if reset_mask.any():
        robot: Entity = env.scene["robot"]
        base_height = robot.data.root_link_pos_w[:, 2]
        # Match HMV: ground-level starts stay forward.  Side-looking is only
        # enabled after the robot has encountered a rough section; elevated
        # starts may use side-looking immediately.
        reset_keep_forward = (base_height < 1.2) | (base_height > 6.0)
        env._h2_yaw_prev_front_flat[reset_mask] = gate[reset_mask] < 0.5
        env._h2_yaw_flat_ready[reset_mask] = ~reset_keep_forward[reset_mask]

    _, _, depth_diff = _camera_depth_lr(env, terrain_camera_name)
    candidate = torch.zeros_like(target)
    flat = gate < 0.5
    candidate[flat & (depth_diff > depth_diff_threshold)] = 1
    candidate[flat & (depth_diff < -depth_diff_threshold)] = -1

    rough = ~flat
    target[rough] = 0
    pending[rough] = 0
    pending_count[rough] = 0
    env._h2_yaw_flat_ready[rough] = True

    active = flat & env._h2_yaw_flat_ready
    different = active & (candidate != target)
    undecided = different & (candidate == 0)
    pending[undecided] = 0
    pending_count[undecided] = 0

    decisive = different & (candidate != 0)
    same_pending = decisive & (candidate == pending)
    pending_count[same_pending] += 1
    new_pending = decisive & (candidate != pending)
    pending[new_pending] = candidate[new_pending]
    pending_count[new_pending] = 1

    required = torch.full_like(pending_count, max(1, confirm_steps))
    side_flip = different & (target != 0) & (candidate != 0)
    required[side_flip] = torch.maximum(required[side_flip], torch.full_like(required[side_flip], 5))
    promote = different & (pending_count >= required)
    target[promote] = pending[promote]
    pending_count[promote] = 0

    env._h2_yaw_prev_front_flat = flat.clone()

    env._h2_yaw_target_last_step = step
    return target


def _head_yaw(
    env: ManagerBasedRlEnv,
    yaw_joint_name: str = "head_yaw_joint",
    yaw_joint_sign: float = 1.0,
) -> torch.Tensor:
    """Return canonical head yaw, positive to the robot's left."""
    robot: Entity = env.scene["robot"]
    joint_ids, _ = robot.find_joints((yaw_joint_name,), preserve_order=True)
    return yaw_joint_sign * robot.data.joint_pos[:, joint_ids[0]]


def _navigation_target_head_yaw(
    env: ManagerBasedRlEnv,
    command_name: str,
    max_head_yaw: float = math.radians(50.0),
) -> torch.Tensor:
    """Return the live target bearing in the body frame, clipped to head limits.

    The pose command owns a world-frame navigation target.  Recomputing its
    bearing every step makes the desired relative head angle naturally shrink
    to zero as the body turns toward that target.
    """
    command_term = env.command_manager.get_term(command_name)
    robot: Entity = env.scene["robot"]
    target_vec_w = command_term.pos_command_w - robot.data.root_link_pos_w
    target_heading_w = torch.atan2(target_vec_w[:, 1], target_vec_w[:, 0])
    target_yaw_b = torch.atan2(
        torch.sin(target_heading_w - robot.data.heading_w),
        torch.cos(target_heading_w - robot.data.heading_w),
    )
    target_yaw_b = torch.clamp(target_yaw_b, min=-max_head_yaw, max=max_head_yaw)

    # Random-velocity and standing terrains do not have a meaningful active
    # position target, so their camera target is straight ahead.
    target_dist = torch.linalg.vector_norm(target_vec_w[:, :2], dim=1)
    valid_target = target_dist > command_term.cfg.target_dis_threshold
    valid_target &= ~command_term.random_velocity_indices
    valid_target &= ~command_term.is_standing_env
    return torch.where(valid_target, target_yaw_b, torch.zeros_like(target_yaw_b))


def camera_yaw_position_target_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float = math.radians(12.0),
    max_head_yaw: float = math.radians(50.0),
    yaw_joint_name: str = "head_yaw_joint",
    yaw_joint_sign: float = 1.0,
) -> torch.Tensor:
    """Reward the camera for facing the live navigation target."""
    desired_yaw = _navigation_target_head_yaw(env, command_name, max_head_yaw)
    camera_yaw = _head_yaw(env, yaw_joint_name, yaw_joint_sign)
    error = torch.atan2(torch.sin(camera_yaw - desired_yaw), torch.cos(camera_yaw - desired_yaw))
    return torch.exp(-torch.square(error) / std**2)


def camera_yaw_position_target_error_l2(
    env: ManagerBasedRlEnv,
    command_name: str,
    max_head_yaw: float = math.radians(50.0),
    yaw_joint_name: str = "head_yaw_joint",
    yaw_joint_sign: float = 1.0,
) -> torch.Tensor:
    """Squared camera-to-live-target bearing error."""
    desired_yaw = _navigation_target_head_yaw(env, command_name, max_head_yaw)
    camera_yaw = _head_yaw(env, yaw_joint_name, yaw_joint_sign)
    error = torch.atan2(torch.sin(camera_yaw - desired_yaw), torch.cos(camera_yaw - desired_yaw))
    return torch.square(error)


def delta_yaw_command_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """HMV-shaped yaw-rate reward against the live navigation command."""
    asset: Entity = env.scene[asset_cfg.name]
    target = env.command_manager.get_command(command_name)[:, 2]
    delta = torch.nan_to_num(
        torch.abs(asset.data.root_link_ang_vel_b[:, 2] - target), nan=10.0, posinf=10.0, neginf=10.0
    )
    over = torch.clamp(delta - 0.1, min=0.0, max=0.3)
    penalty = 0.25 * torch.expm1(8.0 * over)
    return torch.where(delta <= 0.1, torch.full_like(delta, 0.05), -penalty)


def _dynamic_wz_target(
    env: ManagerBasedRlEnv,
    command_name: str,
    head_yaw_threshold: float = math.radians(15.0),
    wz_scale: float = 2.0,
    wz_max: float = 0.8,
    yaw_joint_name: str = "head_yaw_joint",
    yaw_joint_sign: float = 1.0,
    **target_view_kwargs,
) -> torch.Tensor:
    command_wz = env.command_manager.get_command(command_name)[:, 2]
    head_yaw = _head_yaw(env, yaw_joint_name, yaw_joint_sign)
    target_view = _update_depth_target_view(env, **target_view_kwargs)
    extra_target = torch.sign(head_yaw) * torch.clamp(
        wz_scale * (torch.abs(head_yaw) - head_yaw_threshold), min=0.0, max=wz_max
    )
    return torch.where((target_view != 0) & (torch.abs(head_yaw) > head_yaw_threshold), extra_target, command_wz)


def track_ang_vel_z_dynamic_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    **target_view_kwargs,
) -> torch.Tensor:
    """Track command yaw, or follow the physical head-yaw direction at a side opening."""
    asset: Entity = env.scene[asset_cfg.name]
    target = _dynamic_wz_target(env, command_name, **target_view_kwargs)
    error = torch.square(target - asset.data.root_link_ang_vel_b[:, 2])
    return torch.exp(-error / std**2)


def delta_yaw_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    **target_view_kwargs,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    target = _dynamic_wz_target(env, command_name, **target_view_kwargs)
    delta = torch.nan_to_num(
        torch.abs(asset.data.root_link_ang_vel_b[:, 2] - target), nan=10.0, posinf=10.0, neginf=10.0
    )
    over = torch.clamp(delta - 0.1, min=0.0, max=0.3)
    penalty = 0.25 * torch.expm1(8.0 * over)
    return torch.where(delta <= 0.1, torch.full_like(delta, 0.05), -penalty)


def camera_yaw_target_reward(
    env: ManagerBasedRlEnv,
    target_yaw_abs: float = math.pi / 4.0,
    side_yaw_start_abs: float = math.radians(15.0),
    forward_yaw_abs_threshold: float = math.radians(10.0),
    yaw_joint_name: str = "head_yaw_joint",
    yaw_joint_sign: float = 1.0,
    **target_view_kwargs,
) -> torch.Tensor:
    target_view = _update_depth_target_view(env, **target_view_kwargs)
    yaw = _head_yaw(env, yaw_joint_name, yaw_joint_sign)
    forward_reward = torch.clamp(1.0 - torch.abs(yaw) / forward_yaw_abs_threshold, min=0.0)
    # H2 head yaw uses +Z: positive joint yaw is left, while target_view +1 is right.
    yaw_in_target_dir = -target_view.to(yaw.dtype) * yaw
    denom = max(target_yaw_abs - side_yaw_start_abs, 1e-6)
    side_reward = torch.clamp((yaw_in_target_dir - side_yaw_start_abs) / denom, min=0.0, max=1.0)
    return torch.where(target_view == 0, forward_reward, side_reward)


def camera_yaw_side_start_penalty(
    env: ManagerBasedRlEnv,
    side_yaw_start_abs: float = math.radians(15.0),
    yaw_joint_name: str = "head_yaw_joint",
    yaw_joint_sign: float = 1.0,
    **target_view_kwargs,
) -> torch.Tensor:
    target_view = _update_depth_target_view(env, **target_view_kwargs)
    yaw = _head_yaw(env, yaw_joint_name, yaw_joint_sign)
    yaw_in_target_dir = -target_view.to(yaw.dtype) * yaw
    penalty = torch.clamp((side_yaw_start_abs - yaw_in_target_dir) / (2.0 * side_yaw_start_abs), min=0.0)
    return torch.where(target_view == 0, torch.zeros_like(penalty), penalty)


def head_body_yaw_l2(
    env: ManagerBasedRlEnv,
    forward_penalty_scale: float = 10.0,
    yaw_joint_name: str = "head_yaw_joint",
    yaw_joint_sign: float = 1.0,
    **target_view_kwargs,
) -> torch.Tensor:
    target_view = _update_depth_target_view(env, **target_view_kwargs)
    yaw = _head_yaw(env, yaw_joint_name, yaw_joint_sign)
    scale = torch.where(target_view == 0, torch.full_like(yaw, forward_penalty_scale), torch.ones_like(yaw))
    return torch.square(yaw) * scale


def camera_pitch_target_reward(
    env: ManagerBasedRlEnv,
    target_pitch: float,
    std: float,
    rough: bool,
    pitch_joint_name: str = "head_pitch_joint",
    scanner_name: str = "height_scanner",
    rough_threshold: float = 0.05,
    rough_upper_threshold: float = 0.5,
) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    joint_ids, _ = robot.find_joints((pitch_joint_name,), preserve_order=True)
    pitch = robot.data.joint_pos[:, joint_ids[0]]
    gate = roughness_gate_from_scanner(
        env,
        sensor_name=scanner_name,
        rough_threshold=rough_threshold,
        rough_upper_threshold=rough_upper_threshold,
    )
    score = torch.exp(-torch.square(pitch - target_pitch) / std**2)
    return (gate if rough else 1.0 - gate) * score


class camera_axis_action_acc_l2(ManagerTermBase):
    """Penalize second-order action changes on one of the final two head axes."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self._previous = torch.zeros(env.num_envs, device=env.device)
        self._previous_rate = torch.zeros_like(self._previous)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        index = slice(None) if env_ids is None else env_ids
        self._previous[index] = 0.0
        self._previous_rate[index] = 0.0

    def __call__(self, env: ManagerBasedRlEnv, axis_id: int, camera_action_dim: int = 2) -> torch.Tensor:
        action = env.action_manager.action[:, -camera_action_dim + axis_id]
        rate = action - self._previous
        acceleration = rate - self._previous_rate
        self._previous[:] = action
        self._previous_rate[:] = rate
        return torch.square(acceleration)


def heading_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Compute heading command magnitude."""
    command = env.command_manager.get_command(command_name)
    return torch.abs(command[:, 2])


def dont_wait(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize standing still when there is a forward velocity command."""
    asset: Entity = env.scene[asset_cfg.name]
    lin_vel_cmd_x = env.command_manager.get_command(command_name)[:, 0]
    lin_vel_x = asset.data.root_link_lin_vel_b[:, 0]

    return (lin_vel_cmd_x > 0.3) * (
        (lin_vel_x < 0.15).float() + (lin_vel_x < 0.0).float() + (lin_vel_x < -0.15).float()
    )


class delayed_stuck_penalty(ManagerTermBase):
    """Penalize sustained failure to approach the active navigation waypoint.

    Progress is measured in world-frame XY over a fixed window.  The delayed
    decision avoids treating a brief pause for turning or stepping as a stuck
    state.  A waypoint change starts a fresh window so ordered corridor goals
    are evaluated independently.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self._reference_distance = torch.zeros(env.num_envs, device=env.device)
        self._reference_target = torch.zeros(env.num_envs, 3, device=env.device)
        self._elapsed_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._is_stuck = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        index = slice(None) if env_ids is None else env_ids
        self._reference_distance[index] = 0.0
        self._reference_target[index] = 0.0
        self._elapsed_steps[index] = 0
        self._is_stuck[index] = False
        self._initialized[index] = False

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        window_s: float = 1.0,
        min_progress: float = 0.04,
        command_threshold: float = 0.2,
    ) -> torch.Tensor:
        command_term = env.command_manager.get_term(command_name)
        robot: Entity = env.scene["robot"]
        target = command_term.pos_command_w
        distance = torch.linalg.vector_norm(target[:, :2] - robot.data.root_link_pos_w[:, :2], dim=1)

        active = distance > command_term.cfg.target_dis_threshold
        active &= torch.linalg.vector_norm(command_term.vel_command_b[:, :2], dim=1) > command_threshold
        active &= ~command_term.random_velocity_indices
        active &= ~command_term.is_standing_env

        target_changed = torch.linalg.vector_norm(target - self._reference_target, dim=1) > 1.0e-4
        restart = ~self._initialized | target_changed | ~active | (env.episode_length_buf <= 1)
        if torch.any(restart):
            self._reference_distance[restart] = distance[restart]
            self._reference_target[restart] = target[restart]
            self._elapsed_steps[restart] = 0
            self._is_stuck[restart] = False
            self._initialized[restart] = True

        tracking = active & ~restart
        self._elapsed_steps[tracking] += 1
        window_steps = max(1, int(round(window_s / env.step_dt)))
        evaluate = tracking & (self._elapsed_steps >= window_steps)
        if torch.any(evaluate):
            progress = self._reference_distance[evaluate] - distance[evaluate]
            self._is_stuck[evaluate] = progress < min_progress
            self._reference_distance[evaluate] = distance[evaluate]
            self._reference_target[evaluate] = target[evaluate]
            self._elapsed_steps[evaluate] = 0

        return (active & self._is_stuck).float()


def corridor_target_height_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    foot_height_offset: float = 0.035,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=("left_ankle_roll_link", "right_ankle_roll_link")
    ),
) -> torch.Tensor:
    """Reward the feet for approaching the active corridor waypoint height.

    Ordered waypoints are only present on corridor terrains.  The ankle-link
    origins sit above the soles, so ``foot_height_offset`` converts their mean
    height to an approximate sole-plane height before comparing it with the
    platform target Z.
    """
    command_term = env.command_manager.get_term(command_name)
    robot: Entity = env.scene[asset_cfg.name]
    mean_sole_height = (
        torch.mean(robot.data.body_link_pos_w[:, asset_cfg.body_ids, 2], dim=1) - foot_height_offset
    )
    height_error = torch.abs(mean_sole_height - command_term.pos_command_w[:, 2])
    corridor_mask = command_term.ordered_waypoint_count > 0
    return corridor_mask.float() * torch.exp(-height_error / std)


def stand_still(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.15,
    offset: float = 1.0,
) -> torch.Tensor:
    """Penalize moving when there is no velocity command."""
    asset: Entity = env.scene[asset_cfg.name]
    dof_error = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)

    cmd = env.command_manager.get_command(command_name)
    cmd_lin_norm = torch.norm(cmd[:, :2], dim=1)
    cmd_yaw_abs = torch.abs(cmd[:, 2])

    return (dof_error - offset) * (cmd_lin_norm < threshold) * (cmd_yaw_abs < threshold)


def feet_air_time(
    env: ManagerBasedRlEnv,
    command_name: str,
    vel_threshold: float,
    sensor_name: str,
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold
    and also keeping one foot at a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step),
    then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene[sensor_name]
    air_time = contact_sensor.data.current_air_time
    contact_time = contact_sensor.data.current_contact_time
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]

    # no reward for zero command
    cmd = env.command_manager.get_command(command_name)
    reward *= torch.logical_or(
        torch.norm(cmd[:, :2], dim=1) > vel_threshold,
        torch.abs(cmd[:, 2]) > vel_threshold,
    )
    return reward


def feet_slide(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize foot sliding speed while feet are in contact."""
    asset: Entity = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene[sensor_name]

    in_contact = torch.max(torch.linalg.vector_norm(sensor.data.force_history, dim=-1), dim=2)[0] > threshold
    foot_vel_xy = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2]
    slip_speed = torch.norm(foot_vel_xy, dim=-1)
    return torch.sum(slip_speed * in_contact.float(), dim=1)


def ang_vel_xy_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def joint_deviation_square(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_error), dim=1)


def joint_deviation_l1(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(joint_error), dim=1)


def link_orientation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize non-flat link orientation using L2 squared kernel."""
    asset: Entity = env.scene[asset_cfg.name]
    link_quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0], :]
    link_projected_gravity = quat_apply_inverse(link_quat, asset.data.gravity_vec_w)
    return torch.sum(torch.square(link_projected_gravity[:, :2]), dim=1)


def feet_orientation_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward feet being oriented vertically when in contact with the ground."""
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]

    body_link_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    num_envs, num_feet = body_link_quat_w.shape[:2]

    gravity_w = asset.data.gravity_vec_w.unsqueeze(1).expand(-1, num_feet, -1)
    projected_gravity = quat_apply_inverse(body_link_quat_w.reshape(-1, 4), gravity_w.reshape(-1, 3)).reshape(
        num_envs, num_feet, 3
    )
    orientation_error = torch.linalg.vector_norm(projected_gravity[:, :, :2], dim=-1)

    in_contact = (
        torch.max(torch.linalg.vector_norm(contact_sensor.data.force_history, dim=-1), dim=2)[0]
        > contact_force_threshold
    )

    return torch.sum(orientation_error * in_contact.float(), dim=1)


def feet_at_plane(
    env: ManagerBasedRlEnv,
    contact_sensor_name: str,
    left_height_scanner_name: str,
    right_height_scanner_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_offset: float = 0.035,
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward feet being at certain height above the ground plane."""
    asset: Entity = env.scene[asset_cfg.name]
    body_link_pos_w = asset.data.body_link_pos_w

    contact_sensor: ContactSensor = env.scene[contact_sensor_name]
    is_contact = (
        torch.max(torch.linalg.vector_norm(contact_sensor.data.force_history, dim=-1), dim=2)[0]
        > contact_force_threshold
    )

    left_sensor: RayCastSensor = env.scene[left_height_scanner_name]
    right_sensor: RayCastSensor = env.scene[right_height_scanner_name]
    left_hit_z = left_sensor.data.hit_pos_w[..., 2]
    right_hit_z = right_sensor.data.hit_pos_w[..., 2]
    left_hit_z = torch.where(left_sensor.data.distances < 0.0, 0.0, left_hit_z)
    right_hit_z = torch.where(right_sensor.data.distances < 0.0, 0.0, right_hit_z)

    left_height = body_link_pos_w[:, asset_cfg.body_ids[0], 2].unsqueeze(-1)
    right_height = body_link_pos_w[:, asset_cfg.body_ids[1], 2].unsqueeze(-1)

    left_contact = is_contact[:, 0:1].float()
    right_contact = is_contact[:, 1:2].float()

    left_reward = torch.clamp(left_height - left_hit_z - height_offset, min=0.0, max=0.3) * left_contact
    right_reward = torch.clamp(right_height - right_hit_z - height_offset, min=0.0, max=0.3) * right_contact
    return torch.sum(left_reward, dim=-1) + torch.sum(right_reward, dim=-1)


def feet_close_xy_gauss(
    env: ManagerBasedRlEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.1,
) -> torch.Tensor:
    """Penalize when feet are too close together in the y distance."""
    asset: Entity = env.scene[asset_cfg.name]
    body_link_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]

    left_foot_xy = body_link_pos_w[:, 0, :2]
    right_foot_xy = body_link_pos_w[:, 1, :2]
    heading_w = asset.data.heading_w

    cos_heading = torch.cos(heading_w)
    sin_heading = torch.sin(heading_w)

    left_y = -sin_heading * left_foot_xy[:, 0] + cos_heading * left_foot_xy[:, 1]
    right_y = -sin_heading * right_foot_xy[:, 0] + cos_heading * right_foot_xy[:, 1]
    feet_distance_y = torch.abs(left_y - right_y)

    return torch.exp(-torch.clamp(threshold - feet_distance_y, min=0.0) / std**2) - 1


def volume_points_penetration(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    tolerance: float = 0.0,
) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_name]
    penetration = sensor.data.penetration_offset
    points_vel = sensor.data.points_vel_w

    penetration_depth = torch.linalg.vector_norm(penetration.reshape(env.num_envs, -1, 3), dim=-1)
    in_obstacle = (penetration_depth > tolerance).float()
    points_vel_norm = torch.linalg.vector_norm(points_vel.reshape(env.num_envs, -1, 3), dim=-1)
    velocity_times_penetration = in_obstacle * (points_vel_norm + 1e-6) * penetration_depth
    return torch.sum(velocity_times_penetration, dim=-1)


def motors_power_square(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    normalize_by_stiffness: bool = True,
    normalize_by_num_joints: bool = False,
) -> torch.Tensor:
    return _motors_power_square_general(
        env=env,
        asset_cfg=asset_cfg,
        normalize_by_stiffness=normalize_by_stiffness,
        normalize_by_num_joints=normalize_by_num_joints,
    )


def applied_torque_limits_by_ratio(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    limit_ratio: float = 0.8,
) -> torch.Tensor:
    """Penalize when the applied torque exceeds a ratio of torque limits."""
    return _applied_torque_limits_by_ratio_general(
        env=env,
        asset_cfg=asset_cfg,
        limit_ratio=limit_ratio,
    )


class joint_vel_limits:
    """Penalize joints exceeding their soft velocity limits.

    The source limits come from URDF metadata that has no MJCF field. Resolve
    the configured name-to-limit mapping once, following mjlab's native
    class-based reward-term pattern.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
        _, _, velocity_limits = resolve_matching_names_values(
            data=cfg.params["velocity_limits"],
            list_of_strings=joint_names,
        )
        self.velocity_limits = torch.tensor(velocity_limits, device=env.device, dtype=torch.float32)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        soft_ratio: float,
        velocity_limits: dict[str, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        del velocity_limits  # Resolved once in __init__.
        asset: Entity = env.scene[asset_cfg.name]
        out_of_limits = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) - self.velocity_limits * soft_ratio
        return torch.sum(torch.clamp(out_of_limits, min=0.0, max=1.0), dim=1)


def undesired_contacts(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold: float,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    is_contact = torch.max(torch.linalg.vector_norm(contact_sensor.data.force_history, dim=-1), dim=2)[0] > threshold
    return torch.sum(is_contact.float(), dim=1)
