"""Per-environment timed sensor dropout and separately tracked history validity."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class VisualMaskConfig:
    enabled: bool = True
    front_center_deg: float = 0.0
    enter_deg: float = 15.0
    exit_deg: float = 20.0
    min_duration_s: float = 0.2
    max_duration_s: float = 2.0
    max_episode_ratio: float = 0.5
    trigger_rate_hz: float = 0.3
    cooldown_s: float = 0.5

    def validate(self):
        if not 0 <= self.enter_deg < self.exit_deg <= 180:
            raise ValueError("Expected 0 <= enter < exit <= 180 degrees")
        if not 0 < self.min_duration_s <= self.max_duration_s:
            raise ValueError("Invalid mask duration")
        if not 0 <= self.max_episode_ratio <= 0.5 or self.trigger_rate_hz < 0 or self.cooldown_s < 0:
            raise ValueError("Invalid ratio/rate/cooldown")


class HeadYawVisualMask:
    def __init__(self, cfg: VisualMaskConfig, num_envs: int, dt: float, device="cpu", seed=42):
        cfg.validate()
        self.cfg, self.dt = cfg, dt
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.front_view_state = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.mask_active = torch.zeros_like(self.front_view_state)
        for name in ("mask_remaining_time", "mask_elapsed_time", "mask_total_duration",
                     "mask_cooldown", "episode_elapsed_time"):
            setattr(self, name, torch.zeros(num_envs, device=device))

    def reset(self, done):
        for name in ("front_view_state", "mask_active", "mask_remaining_time", "mask_elapsed_time",
                     "mask_total_duration", "mask_cooldown", "episode_elapsed_time"):
            getattr(self, name)[done.bool()] = 0

    def step(self, relative_yaw: torch.Tensor):
        c, dt = self.cfg, self.dt
        error = relative_yaw - math.radians(c.front_center_deg)
        error = torch.atan2(error.sin(), error.cos()).abs()
        self.front_view_state = torch.where(error <= math.radians(c.enter_deg), True,
                                           torch.where(error >= math.radians(c.exit_deg), False,
                                                       self.front_view_state))
        self.episode_elapsed_time += dt
        self.mask_cooldown.sub_(dt).clamp_(min=0)
        # Prefix budget guarantees the ratio even when an episode ends early.
        credit = c.max_episode_ratio * self.episode_elapsed_time - self.mask_total_duration
        eligible = ~self.mask_active & self.front_view_state & (self.mask_cooldown <= 0)
        eligible &= credit >= c.min_duration_s
        event = torch.rand(error.shape, device=error.device, generator=self.generator)
        trigger = eligible & (event < -math.expm1(-c.trigger_rate_hz * dt)) & c.enabled
        duration = c.min_duration_s + torch.rand(error.shape, device=error.device,
                                                generator=self.generator) * (c.max_duration_s - c.min_duration_s)
        # Reserve enough earned budget to finish the sampled dropout rather
        # than silently breaking a long mask into one-step events.
        duration = torch.minimum(duration, credit.clamp(min=0))
        self.mask_remaining_time[trigger] = duration[trigger]
        self.mask_elapsed_time[trigger] = 0
        self.mask_active |= trigger
        ended = self.mask_active & (~self.front_view_state | (self.mask_remaining_time < dt - 1e-7)
                                    | (credit < dt - 1e-7) | (not c.enabled))
        self.mask_active[ended] = False
        self.mask_remaining_time[ended] = 0
        self.mask_cooldown[ended] = c.cooldown_s
        self.mask_total_duration += self.mask_active.float() * dt
        self.mask_elapsed_time += self.mask_active.float() * dt
        self.mask_remaining_time.sub_(self.mask_active.float() * dt).clamp_(min=0)
        return ~self.mask_active

    def state_dict(self):
        return {**{k: v.clone() for k, v in vars(self).items() if isinstance(v, torch.Tensor)},
                "generator_state": self.generator.get_state()}

    def load_state_dict(self, state):
        for k, v in state.items():
            if k == "generator_state":
                self.generator.set_state(v.cpu())
            else:
                getattr(self, k).copy_(v)


class DepthValidityHistory:
    """Same raw history length and sampled offsets as delayed_visualizable_image."""
    def __init__(self, num_envs, length=37, num_frames=8, skip=5, device="cpu"):
        self.valid = torch.zeros(num_envs, length, dtype=torch.bool, device=device)
        self.offsets = torch.arange(num_frames - 1, -1, -1, device=device) * skip

    def reset(self, done):
        self.valid[done.bool()] = False

    def append(self, sensor_valid, refreshed=None):
        rolled = torch.cat((self.valid[:, 1:], sensor_valid.bool()[:, None]), dim=1)
        self.valid = rolled if refreshed is None else torch.where(refreshed[:, None], rolled, self.valid)

    def sample(self, delay):
        indices = self.valid.shape[1] - 1 - self.offsets[None] - delay.long()[:, None]
        if (indices < 0).any():
            raise ValueError("Depth validity history is shorter than the selected window")
        return self.valid.gather(1, indices)
