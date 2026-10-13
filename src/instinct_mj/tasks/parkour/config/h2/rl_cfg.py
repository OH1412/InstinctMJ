"""PPO runner configuration for the Unitree H2 Parkour task."""

from __future__ import annotations

from dataclasses import dataclass, field

from instinct_mj.rl import (
    InstinctRlConv2dHeadCfg,
    InstinctRlEncoderMoEActorCriticCfg,
    InstinctRlNormalizerCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)


@dataclass(kw_only=True)
class H2DepthEncoderConv2dCfg(InstinctRlConv2dHeadCfg):
    output_size: int = 128
    channels: list[int] = field(default_factory=lambda: [16, 32, 32])
    kernel_sizes: list[int] = field(default_factory=lambda: [5, 3, 3])
    strides: list[int] = field(default_factory=lambda: [2, 2, 1])
    paddings: list[int] = field(default_factory=lambda: [2, 1, 1])
    hidden_sizes: list[int] = field(default_factory=lambda: [256])
    nonlinearity: str = "ReLU"
    use_maxpool: bool = False
    component_names: list[str] = field(default_factory=lambda: ["depth_image"])
    takeout_input_components: bool = True


@dataclass(kw_only=True)
class H2EncoderConfigs:
    depth_encoder: object = field(default_factory=H2DepthEncoderConv2dCfg)


@dataclass(kw_only=True)
class H2PolicyCfg(InstinctRlEncoderMoEActorCriticCfg):
    init_noise_std: float = 1.0
    num_moe_experts: int = 4
    actor_hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    activation: str = "elu"
    encoder_configs: object = field(default_factory=H2EncoderConfigs)
    critic_encoder_configs: object = field(default_factory=H2EncoderConfigs)


@dataclass(kw_only=True)
class H2PpoAlgorithmCfg(InstinctRlPpoAlgorithmCfg):
    class_name: str = "PPO"
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.006
    learning_rate: float = 1.0e-3
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0


@dataclass(kw_only=True)
class H2ParkourPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 24
    policy_observation_group: str = "policy"
    critic_observation_group: str = "critic"
    max_iterations: int = 300_000
    save_interval: int = 500
    log_interval: int = 10
    experiment_name: str = "h2_parkour_depth_ppo"
    resume: bool = False
    load_run: str | None = None
    policy: object = field(default_factory=H2PolicyCfg)
    algorithm: object = field(default_factory=H2PpoAlgorithmCfg)
    normalizers: dict[str, InstinctRlNormalizerCfg] = field(
        default_factory=lambda: {
            "policy": InstinctRlNormalizerCfg(),
            "critic": InstinctRlNormalizerCfg(),
        }
    )


def h2_parkour_instinct_rl_cfg() -> H2ParkourPPORunnerCfg:
    return H2ParkourPPORunnerCfg()
