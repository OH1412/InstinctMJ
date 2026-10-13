"""Action-controlled depth-camera view for an additive parkour experiment.

The camera remains attached to ``torso_link``.  Only its local yaw and pitch
offset are changed; no robot joint or head actuator is involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api import math as math_utils

from instinct_mj.sensors.noisy_camera import NoisyGroupedRayCasterCamera

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class ActionableNoisyGroupedRayCasterCamera(NoisyGroupedRayCasterCamera):
    """Noisy grouped ray-caster camera with per-environment yaw/pitch offsets."""

    def initialize(self, mj_model, model, data, device: str) -> None:
        super().initialize(mj_model, model, data, device)
        self._nominal_offset_quat = self._offset_quat.clone()
        self._view_yaw = torch.zeros(self._num_envs, device=device)
        self._view_pitch = torch.zeros(self._num_envs, device=device)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = self._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self._runtime_device, dtype=torch.long)
        self._view_yaw[env_ids] = 0.0
        self._view_pitch[env_ids] = 0.0
        self._offset_quat[env_ids] = self._nominal_offset_quat[env_ids]

    def set_view_angles(self, yaw: torch.Tensor, pitch: torch.Tensor) -> None:
        """Set local camera yaw/pitch in radians for every environment."""
        if self._offset_quat is None or self._nominal_offset_quat is None:
            raise RuntimeError("camera must be initialized before setting view angles")
        yaw = yaw.to(device=self._offset_quat.device, dtype=self._offset_quat.dtype)
        pitch = pitch.to(device=self._offset_quat.device, dtype=self._offset_quat.dtype)
        if yaw.shape != (self._num_envs,) or pitch.shape != (self._num_envs,):
            raise ValueError(
                f"camera view angles must have shape ({self._num_envs},), got {tuple(yaw.shape)} and {tuple(pitch.shape)}"
            )

        self._view_yaw[:] = yaw
        self._view_pitch[:] = pitch
        zeros = torch.zeros_like(yaw)
        # Camera convention uses +X forward, +Y left, +Z up.  Apply pitch
        # around local Y and yaw around local Z, then compose with nominal pose.
        delta_quat = math_utils.quat_from_euler_xyz(zeros, pitch, yaw)
        self._offset_quat[:] = math_utils.quat_mul(delta_quat, self._nominal_offset_quat)
        self._has_fresh_sense = False


class CameraViewAction(ActionTerm):
    """Two-dimensional absolute camera-view action, normalized to [-1, 1]."""

    cfg: "CameraViewActionCfg"

    def __init__(self, cfg: "CameraViewActionCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._sensor = env.scene.sensors[cfg.camera_name]
        if not isinstance(self._sensor, ActionableNoisyGroupedRayCasterCamera):
            raise TypeError(
                f"camera '{cfg.camera_name}' must use ActionableNoisyGroupedRayCasterCamera, "
                f"got {type(self._sensor).__name__}"
            )

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = torch.clamp(actions, -1.0, 1.0)
        yaw = self._raw_actions[:, 0] * self.cfg.yaw_limit_rad
        pitch = self._raw_actions[:, 1] * self.cfg.pitch_limit_rad
        self._sensor.set_view_angles(yaw, pitch)

    def apply_actions(self) -> None:
        # Camera rays are prepared by the sensor during the normal simulation
        # sense phase.  There is no MuJoCo actuator to apply here.
        return None

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
            env_ids = slice(None)
        else:
            self._raw_actions[env_ids] = 0.0
        if isinstance(env_ids, slice):
            yaw = torch.zeros(self.num_envs, device=self.device)
            pitch = torch.zeros(self.num_envs, device=self.device)
        else:
            yaw = self._sensor._view_yaw.clone()
            pitch = self._sensor._view_pitch.clone()
            yaw[env_ids] = 0.0
            pitch[env_ids] = 0.0
        self._sensor.set_view_angles(yaw, pitch)


@dataclass(kw_only=True)
class CameraViewActionCfg(ActionTermCfg):
    """Configuration for the two extra camera-view action dimensions."""

    entity_name: str = "robot"
    camera_name: str = "camera"
    yaw_limit_rad: float = 0.7853981633974483  # +/-45 degrees
    pitch_limit_rad: float = 0.5235987755982988  # +/-30 degrees
    class_type: type[ActionTerm] = CameraViewAction

    def build(self, env: ManagerBasedRlEnv) -> ActionTerm:
        return self.class_type(self, env)

