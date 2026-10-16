"""Register Instinct Mj parkour G1 tasks."""

# Copyright (c) 2022-2025, The Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from instinct_mj.tasks.registry import register_instinct_task


def _parkour_amp_env_cfg(play: bool, shoe: bool = True):
    from .g1_parkour_target_amp_cfg import instinct_g1_parkour_amp_final_cfg

    return instinct_g1_parkour_amp_final_cfg(play=play, shoe=shoe)


def _parkour_camera_view_env_cfg(play: bool, shoe: bool = True):
    from .camera_view_env_cfg import instinct_g1_parkour_camera_view_env_cfg

    return instinct_g1_parkour_camera_view_env_cfg(play=play, shoe=shoe)


def g1_parkour_amp_instinct_rl_cfg():
    from .rl_cfg import g1_parkour_amp_instinct_rl_cfg as _cfg

    return _cfg()


def _parkour_d455_amp_env_cfg(play: bool):
    from .g1_d455_parkour_cfg import instinct_g1_d455_parkour_amp_cfg

    return instinct_g1_d455_parkour_amp_cfg(play=play)


def _parkour_d455_legacy_terrain_amp_env_cfg(play: bool):
    from .g1_d455_parkour_cfg import instinct_g1_d455_parkour_amp_legacy_terrain_cfg

    return instinct_g1_d455_parkour_amp_legacy_terrain_cfg(play=play)


def _parkour_d455_legacy_hmv_terrain_amp_env_cfg(play: bool):
    from .g1_d455_parkour_cfg import instinct_g1_d455_parkour_amp_legacy_hmv_terrain_cfg

    return instinct_g1_d455_parkour_amp_legacy_hmv_terrain_cfg(play=play)


def _parkour_d455_spiral_terrain_amp_env_cfg(play: bool):
    from .g1_d455_parkour_cfg import instinct_g1_d455_parkour_amp_spiral_terrain_cfg

    return instinct_g1_d455_parkour_amp_spiral_terrain_cfg(play=play)


def g1_d455_parkour_amp_instinct_rl_cfg():
    from .rl_cfg import g1_d455_parkour_amp_instinct_rl_cfg as _cfg

    return _cfg()


def g1_parkour_amp_transformer_instinct_rl_cfg():
    from .rl_cfg_transformer import g1_parkour_amp_transformer_instinct_rl_cfg as _cfg

    return _cfg()


def g1_parkour_amp_saliency_gated_instinct_rl_cfg():
    """Configuration for the additive saliency-gated visual experiment."""
    from .rl_cfg_saliency_gated import g1_parkour_amp_saliency_gated_instinct_rl_cfg as _cfg

    return _cfg()


def g1_parkour_amp_camera_view_instinct_rl_cfg():
    """Configuration for the additive action-controlled camera experiment."""
    from .rl_cfg_camera_view import g1_parkour_amp_camera_view_instinct_rl_cfg as _cfg

    return _cfg()


register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-G1-v0",
    env_cfg_factory=lambda: _parkour_amp_env_cfg(play=False, shoe=True),
    play_env_cfg_factory=lambda: _parkour_amp_env_cfg(play=True, shoe=True),
    instinct_rl_cfg_factory=g1_parkour_amp_instinct_rl_cfg,
)


register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-G1-D455-v0",
    env_cfg_factory=lambda: _parkour_d455_amp_env_cfg(play=False),
    play_env_cfg_factory=lambda: _parkour_d455_amp_env_cfg(play=True),
    instinct_rl_cfg_factory=g1_d455_parkour_amp_instinct_rl_cfg,
)


register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-G1-D455-LegacyTerrain-v0",
    env_cfg_factory=lambda: _parkour_d455_legacy_terrain_amp_env_cfg(play=False),
    play_env_cfg_factory=lambda: _parkour_d455_legacy_terrain_amp_env_cfg(play=True),
    instinct_rl_cfg_factory=g1_d455_parkour_amp_instinct_rl_cfg,
)


register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-G1-D455-LegacyHmvTerrain-v0",
    env_cfg_factory=lambda: _parkour_d455_legacy_hmv_terrain_amp_env_cfg(play=False),
    play_env_cfg_factory=lambda: _parkour_d455_legacy_hmv_terrain_amp_env_cfg(play=True),
    instinct_rl_cfg_factory=g1_d455_parkour_amp_instinct_rl_cfg,
)


register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-G1-D455-SpiralTerrain-v0",
    env_cfg_factory=lambda: _parkour_d455_spiral_terrain_amp_env_cfg(play=False),
    play_env_cfg_factory=lambda: _parkour_d455_spiral_terrain_amp_env_cfg(play=True),
    instinct_rl_cfg_factory=g1_d455_parkour_amp_instinct_rl_cfg,
)


register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-G1-D455-Play-v0",
    env_cfg_factory=lambda: _parkour_d455_amp_env_cfg(play=True),
    play_env_cfg_factory=lambda: _parkour_d455_amp_env_cfg(play=True),
    instinct_rl_cfg_factory=g1_d455_parkour_amp_instinct_rl_cfg,
)


register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-G1-Transformer-v0",
    env_cfg_factory=lambda: _parkour_amp_env_cfg(play=False, shoe=True),
    play_env_cfg_factory=lambda: _parkour_amp_env_cfg(play=True, shoe=True),
    instinct_rl_cfg_factory=g1_parkour_amp_transformer_instinct_rl_cfg,
)


register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-G1-Play-v0",
    env_cfg_factory=lambda: _parkour_amp_env_cfg(play=True, shoe=True),
    play_env_cfg_factory=lambda: _parkour_amp_env_cfg(play=True, shoe=True),
    instinct_rl_cfg_factory=g1_parkour_amp_instinct_rl_cfg,
)


# Additive experiment: the original task IDs above remain unchanged.
register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-G1-SaliencyGated-v0",
    env_cfg_factory=lambda: _parkour_amp_env_cfg(play=False, shoe=True),
    play_env_cfg_factory=lambda: _parkour_amp_env_cfg(play=True, shoe=True),
    instinct_rl_cfg_factory=g1_parkour_amp_saliency_gated_instinct_rl_cfg,
)


register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-G1-SaliencyGated-Play-v0",
    env_cfg_factory=lambda: _parkour_amp_env_cfg(play=True, shoe=True),
    play_env_cfg_factory=lambda: _parkour_amp_env_cfg(play=True, shoe=True),
    instinct_rl_cfg_factory=g1_parkour_amp_saliency_gated_instinct_rl_cfg,
)


# Additive experiment: original task IDs and their configs are unchanged.
register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-G1-CameraView-v0",
    env_cfg_factory=lambda: _parkour_camera_view_env_cfg(play=False, shoe=True),
    play_env_cfg_factory=lambda: _parkour_camera_view_env_cfg(play=True, shoe=True),
    instinct_rl_cfg_factory=g1_parkour_amp_camera_view_instinct_rl_cfg,
    runner_entry_point="instinct_mj.rl.camera_view_runner:CameraViewOnPolicyRunner",
)


register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-G1-CameraView-Play-v0",
    env_cfg_factory=lambda: _parkour_camera_view_env_cfg(play=True, shoe=True),
    play_env_cfg_factory=lambda: _parkour_camera_view_env_cfg(play=True, shoe=True),
    instinct_rl_cfg_factory=g1_parkour_amp_camera_view_instinct_rl_cfg,
    runner_entry_point="instinct_mj.rl.camera_view_runner:CameraViewOnPolicyRunner",
)
