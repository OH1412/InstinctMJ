"""Unitree H2 Parkour task registrations."""

from instinct_mj.tasks.registry import register_instinct_task


def _h2_env_cfg(play: bool):
    from .h2_parkour_cfg import instinct_h2_parkour_env_cfg

    return instinct_h2_parkour_env_cfg(play=play)


def h2_parkour_instinct_rl_cfg():
    from .rl_cfg import h2_parkour_instinct_rl_cfg as factory

    return factory()


register_instinct_task(
    task_id="Instinct-Parkour-Target-PPO-H2-v0",
    env_cfg_factory=lambda: _h2_env_cfg(False),
    play_env_cfg_factory=lambda: _h2_env_cfg(True),
    instinct_rl_cfg_factory=h2_parkour_instinct_rl_cfg,
)

register_instinct_task(
    task_id="Instinct-Parkour-Target-PPO-H2-Play-v0",
    env_cfg_factory=lambda: _h2_env_cfg(True),
    play_env_cfg_factory=lambda: _h2_env_cfg(True),
    instinct_rl_cfg_factory=h2_parkour_instinct_rl_cfg,
)
