"""Additive camera-view environment configuration for G1 parkour."""

from __future__ import annotations

import copy

from instinct_mj.tasks.parkour.camera_view_control import (
    ActionableNoisyGroupedRayCasterCamera,
    CameraViewActionCfg,
)

from .g1_parkour_target_amp_cfg import instinct_g1_parkour_amp_final_cfg


def instinct_g1_parkour_camera_view_env_cfg(*, play: bool = False, shoe: bool = True):
    """Return the original parkour env plus two non-articulated camera actions."""
    cfg = instinct_g1_parkour_amp_final_cfg(play=play, shoe=shoe)

    # The factory above returns a fresh config.  Change only this new copy.
    sensors = list(cfg.scene.sensors)
    camera_cfg = next(sensor for sensor in sensors if sensor.name == "camera")
    camera_cfg.class_type = ActionableNoisyGroupedRayCasterCamera
    cfg.scene.sensors = tuple(sensors)

    cfg.actions = dict(cfg.actions)
    cfg.actions["camera_view"] = CameraViewActionCfg()

    # Keep the original policy/critic input dimensionality (8 x 29 actions).
    # The two camera actions are intentionally not fed back as proprioception.
    for group_name in ("policy", "critic"):
        action_term = cfg.observations[group_name].terms["actions"]
        action_term.params = {"action_name": "joint_pos"}

    return cfg

