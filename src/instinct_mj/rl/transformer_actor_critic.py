"""Transformer Actor-Critic variants for experiments.

This module is intentionally separate from the stock instinct_rl policy classes.
The existing ``EncoderMoEActorCritic`` remains untouched and is still the
default policy for the original parkour task.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from instinct_rl.modules.actor_critic import ActorCritic, get_activation
from instinct_rl.modules.encoder_actor_critic import EncoderActorCriticMixin
from instinct_rl.utils.utils import get_subobs_size


class _TokenTransformer(nn.Module):
    """Turn observation components into tokens and aggregate them with attention.

    A component with a temporal shape ``(T, D)`` becomes T tokens. A flat
    component ``(D,)`` becomes one token. Each component has its own linear
    projection because observation components have different widths.
    """

    def __init__(
        self,
        obs_segments: Dict[str, tuple],
        output_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        max_tokens: int | None = None,
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        self.obs_segments = OrderedDict(obs_segments)
        self.d_model = d_model

        self._component_dims: list[tuple[str, int, int]] = []
        projections = OrderedDict()
        total_tokens = 1  # CLS token
        for index, (name, shape) in enumerate(self.obs_segments.items()):
            shape = tuple(int(v) for v in shape)
            if len(shape) == 0:
                raise ValueError(f"Observation component {name!r} has an empty shape")
            sequence_length = shape[0] if len(shape) > 1 else 1
            feature_dim = int(np.prod(shape[1:])) if len(shape) > 1 else shape[0]
            key = f"component_{index}"
            projections[key] = nn.Linear(feature_dim, d_model)
            self._component_dims.append((name, sequence_length, feature_dim))
            total_tokens += sequence_length

        self.projections = nn.ModuleDict(projections)
        self.max_tokens = max_tokens or total_tokens
        if self.max_tokens < total_tokens:
            raise ValueError(f"max_tokens={self.max_tokens} is smaller than required {total_tokens}")
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position_embedding = nn.Parameter(torch.zeros(1, self.max_tokens, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.output = nn.Linear(d_model, output_size)

    def forward(self, flat_observations: torch.Tensor) -> torch.Tensor:
        batch_size = flat_observations.shape[0]
        tokens = [self.cls_token.expand(batch_size, -1, -1)]
        offset = 0
        for index, (name, sequence_length, feature_dim) in enumerate(self._component_dims):
            shape = tuple(int(v) for v in self.obs_segments[name])
            numel = int(np.prod(shape))
            component = flat_observations[:, offset : offset + numel].reshape(batch_size, *shape)
            offset += numel
            if len(shape) == 1:
                component = component.unsqueeze(1)
            else:
                component = component.reshape(batch_size, sequence_length, feature_dim)
            tokens.append(self.projections[f"component_{index}"](component))

        x = torch.cat(tokens, dim=1)
        x = x + self.position_embedding[:, : x.shape[1]]
        x = self.encoder(x)
        return self.output(x[:, 0])


class TransformerActorCritic(ActorCritic):
    """Actor-Critic whose actor and critic backbones are separate Transformers."""

    def __init__(
        self,
        obs_format,
        num_actions,
        actor_hidden_dims=None,
        critic_hidden_dims=None,
        activation="gelu",
        init_noise_std=1.0,
        num_rewards=1,
        transformer_d_model=128,
        transformer_num_heads=4,
        transformer_num_layers=2,
        transformer_dim_feedforward=256,
        transformer_dropout=0.0,
        transformer_max_tokens=None,
        transformer_norm_first=True,
        **kwargs,
    ):
        self._transformer_obs_format = obs_format
        self._transformer_d_model = transformer_d_model
        self._transformer_num_heads = transformer_num_heads
        self._transformer_num_layers = transformer_num_layers
        self._transformer_dim_feedforward = transformer_dim_feedforward
        self._transformer_dropout = transformer_dropout
        self._transformer_max_tokens = transformer_max_tokens
        self._transformer_norm_first = transformer_norm_first
        super().__init__(
            obs_format=obs_format,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims or [],
            critic_hidden_dims=critic_hidden_dims or [],
            activation=activation,
            init_noise_std=init_noise_std,
            num_rewards=num_rewards,
            **kwargs,
        )

    def _build_transformer(self, segments, output_size):
        return _TokenTransformer(
            obs_segments=segments,
            output_size=output_size,
            d_model=self._transformer_d_model,
            num_heads=self._transformer_num_heads,
            num_layers=self._transformer_num_layers,
            dim_feedforward=self._transformer_dim_feedforward,
            dropout=self._transformer_dropout,
            activation="gelu",
            max_tokens=self._transformer_max_tokens,
            norm_first=self._transformer_norm_first,
        )

    def _build_actor(self, num_actions):
        return self._build_transformer(self._transformer_obs_format["policy"], num_actions)

    def _build_critic(self, num_values=1):
        critic_segments = self._transformer_obs_format.get(
            "critic", self._transformer_obs_format["policy"]
        )
        return self._build_transformer(critic_segments, num_values)


class EncoderTransformerActorCritic(EncoderActorCriticMixin, TransformerActorCritic):
    """Depth CNN encoder followed by independent Transformer actor/critic heads."""

    pass

