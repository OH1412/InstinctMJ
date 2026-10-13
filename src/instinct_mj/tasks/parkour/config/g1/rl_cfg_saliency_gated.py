"""Isolated saliency-gated visual policy configuration.

The original ``rl_cfg.py`` is deliberately not imported for mutation and is
not changed by this file.  Only the AMP algorithm settings are reused.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from instinct_mj.rl import InstinctRlEncoderMoEActorCriticCfg, InstinctRlOnPolicyRunnerCfg

from .rl_cfg import AmpAlgoCfg


@dataclass(kw_only=True)
class SaliencyGatedDepthEncoderCfg:
    class_name: str = "SaliencyGatedDepthEncoder"
    component_names: list[str] = field(default_factory=lambda: ["depth_image"])
    output_size: int = 128
    takeout_input_components: bool = True
    channels: list[int] = field(default_factory=lambda: [4])
    kernel_sizes: list[int] = field(default_factory=lambda: [3])
    strides: list[int] = field(default_factory=lambda: [1])
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])
    paddings: list[int] = field(default_factory=lambda: [1])
    nonlinearity: str = "ReLU"
    use_maxpool: bool = True
    gate_init: float = -2.0


def _encoder_configs() -> dict:
    return {
        "class_name": "instinct_mj.rl.saliency_gated_encoder:SaliencyGatedParallelLayer",
        "depth_encoder": SaliencyGatedDepthEncoderCfg(),
    }


@dataclass(kw_only=True)
class SaliencyGatedMoEPolicyCfg(InstinctRlEncoderMoEActorCriticCfg):
    init_noise_std: float = 1.0
    num_moe_experts: int = 4
    actor_hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    activation: str = "elu"
    encoder_configs: object = field(default_factory=_encoder_configs)
    critic_encoder_configs: object = field(default_factory=_encoder_configs)


@dataclass(kw_only=True)
class G1ParkourSaliencyGatedPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 24
    policy_observation_group: str = "policy"
    critic_observation_group: str = "critic"
    max_iterations: int = 300000
    save_interval: int = 5000
    experiment_name: str = "g1_parkour_saliency_gated"
    resume: bool = False
    load_run: str = "^(?!_play$).*"
    empirical_normalization: bool = False
    policy: object = field(default_factory=SaliencyGatedMoEPolicyCfg)
    algorithm: object = field(default_factory=AmpAlgoCfg)


def g1_parkour_amp_saliency_gated_instinct_rl_cfg() -> G1ParkourSaliencyGatedPPORunnerCfg:
    return G1ParkourSaliencyGatedPPORunnerCfg()

