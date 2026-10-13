"""Optional Transformer policy for the G1 parkour AMP task.

The stock ``rl_cfg.py`` remains unchanged. This file is selected only by the
experimental Transformer task registration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from instinct_mj.rl import InstinctRlEncoderActorCriticCfg, InstinctRlOnPolicyRunnerCfg
from .rl_cfg import AmpAlgoCfg, DepthEncoderConv2dCfg


@dataclass(kw_only=True)
class TransformerPolicyCfg(InstinctRlEncoderActorCriticCfg):
    class_name: str = (
        "instinct_mj.rl.transformer_actor_critic:EncoderTransformerActorCritic"
    )
    init_noise_std: float = 1.0
    activation: str = "gelu"
    encoder_configs: dict = field(
        default_factory=lambda: {"depth_encoder": DepthEncoderConv2dCfg()}
    )
    critic_encoder_configs: dict = field(
        default_factory=lambda: {"depth_encoder": DepthEncoderConv2dCfg()}
    )
    transformer_d_model: int = 128
    transformer_num_heads: int = 4
    transformer_num_layers: int = 2
    transformer_dim_feedforward: int = 256
    transformer_dropout: float = 0.0
    transformer_norm_first: bool = True


@dataclass(kw_only=True)
class G1ParkourTransformerPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 24
    policy_observation_group: str = "policy"
    critic_observation_group: str = "critic"
    max_iterations: int = 300000
    save_interval: int = 5000
    experiment_name: str = "g1_parkour_transformer"
    resume: bool = False
    load_run: str = "^(?!_play$).*"
    empirical_normalization: bool = False
    policy: object = field(default_factory=TransformerPolicyCfg)
    algorithm: object = field(default_factory=AmpAlgoCfg)


def g1_parkour_amp_transformer_instinct_rl_cfg() -> G1ParkourTransformerPPORunnerCfg:
    return G1ParkourTransformerPPORunnerCfg()
