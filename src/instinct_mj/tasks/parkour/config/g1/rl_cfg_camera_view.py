"""RL configuration for the additive camera-view action experiment."""

from __future__ import annotations

from dataclasses import dataclass, field

from instinct_mj.rl import InstinctRlOnPolicyRunnerCfg

from .rl_cfg import AmpAlgoCfg, MoEPolicyCfg


@dataclass(kw_only=True)
class G1ParkourCameraViewPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 24
    policy_observation_group: str = "policy"
    critic_observation_group: str = "critic"
    max_iterations: int = 300000
    save_interval: int = 5000
    experiment_name: str = "g1_parkour_camera_view"
    resume: bool = False
    load_run: str = "^(?!_play$).*"
    empirical_normalization: bool = False
    policy: object = field(default_factory=MoEPolicyCfg)
    algorithm: object = field(default_factory=AmpAlgoCfg)


def g1_parkour_amp_camera_view_instinct_rl_cfg() -> G1ParkourCameraViewPPORunnerCfg:
    return G1ParkourCameraViewPPORunnerCfg()

