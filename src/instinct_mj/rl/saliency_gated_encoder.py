"""Saliency-prior visual encoder for the isolated parkour experiment.

This module is intentionally additive.  It does not replace the original
``Conv2dHeadModel`` or modify any of the upstream ``instinct_rl`` modules.

The encoder follows the useful part of the Echo-in-the-Steps design:

* one shared image encoder is applied to every depth-history frame;
* a depth-gradient saliency score supplies an explicit temporal prior;
* a learned gate selects history residuals relative to the current frame.

The input is expected to have shape ``[batch, frames, height, width]`` and
the last frame is treated as the current observation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict

import torch
from torch import nn

from instinct_rl.modules.conv2d import Conv2dHeadModel
from instinct_rl.modules.parallel_layer import ParallelLayer
from instinct_rl.utils.utils import get_subobs_by_components


class SaliencyGatedDepthEncoder(nn.Module):
    """Encode depth history with a saliency-guided gated residual fusion."""

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        output_size: int = 128,
        channels: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        strides: list[int] | None = None,
        hidden_sizes: list[int] | None = None,
        paddings: list[int] | None = None,
        nonlinearity: str = "ReLU",
        use_maxpool: bool = True,
        gate_init: float = -2.0,
        **_: object,
    ):
        super().__init__()
        if len(image_shape) != 3:
            raise ValueError(f"depth image shape must be (frames, height, width), got {image_shape}")

        self.num_frames, self.height, self.width = (int(v) for v in image_shape)
        if self.num_frames < 1:
            raise ValueError("depth history must contain at least one frame")

        channels = [4] if channels is None else list(channels)
        kernel_sizes = [3] * len(channels) if kernel_sizes is None else list(kernel_sizes)
        strides = [1] * len(channels) if strides is None else list(strides)
        paddings = [1] * len(channels) if paddings is None else list(paddings)
        hidden_sizes = [256, 256] if hidden_sizes is None else list(hidden_sizes)

        # The temporal dimension is handled explicitly below.  The CNN always
        # receives one depth image as one input channel, so frames share weights.
        self.frame_encoder = Conv2dHeadModel(
            image_shape=(1, self.height, self.width),
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            paddings=paddings,
            nonlinearity=nonlinearity,
            use_maxpool=use_maxpool,
        )

        gate_hidden = max(32, output_size // 2)
        self.gate_scorer = nn.Sequential(
            nn.Linear(output_size + 1, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
        )
        self.saliency_scale = nn.Parameter(torch.tensor(1.0))
        self.gate_strength = nn.Parameter(torch.tensor(float(gate_init)))
        self.register_buffer(
            "vertical_weights",
            torch.linspace(1.0 / max(self.height - 1, 1), 1.0, max(self.height - 1, 1)).view(1, 1, -1, 1),
            persistent=False,
        )
        if self.num_frames > 1:
            self.history_bias = nn.Parameter(torch.zeros(self.num_frames - 1))
        else:
            self.history_bias = None

    def _saliency(self, depth: torch.Tensor) -> torch.Tensor:
        """Compute a scalar vertical depth-gradient saliency per frame."""
        depth = torch.nan_to_num(depth.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if self.height <= 1:
            return torch.zeros(depth.shape[:2], device=depth.device, dtype=depth.dtype)
        vertical_gradient = (depth[:, :, 1:, :] - depth[:, :, :-1, :]).abs()
        weights = self.vertical_weights.to(device=depth.device, dtype=depth.dtype)
        return (vertical_gradient * weights).mean(dim=(2, 3))

    def forward(self, depth_history: torch.Tensor) -> torch.Tensor:
        if depth_history.ndim != 4:
            raise ValueError(f"expected [batch, frames, height, width], got {tuple(depth_history.shape)}")
        if tuple(depth_history.shape[1:]) != (self.num_frames, self.height, self.width):
            raise ValueError(
                "depth shape mismatch: "
                f"expected {(self.num_frames, self.height, self.width)}, got {tuple(depth_history.shape[1:])}"
            )

        batch_size = depth_history.shape[0]
        frames = torch.nan_to_num(depth_history.float(), nan=0.0, posinf=0.0, neginf=0.0)
        frame_latents = self.frame_encoder(frames.reshape(batch_size * self.num_frames, 1, self.height, self.width))
        frame_latents = frame_latents.reshape(batch_size, self.num_frames, -1)
        current = frame_latents[:, -1]

        if self.num_frames == 1:
            return current

        history = frame_latents[:, :-1]
        saliency = self._saliency(frames)
        relative_saliency = saliency[:, :-1] - saliency[:, -1:].detach()
        relative_saliency = relative_saliency / (relative_saliency.std(dim=1, keepdim=True) + 1.0e-6)

        scorer_input = torch.cat([history, relative_saliency.unsqueeze(-1)], dim=-1)
        scores = self.gate_scorer(scorer_input).squeeze(-1)
        scores = scores + self.saliency_scale * relative_saliency
        scores = scores + self.history_bias.unsqueeze(0)
        weights = torch.softmax(scores, dim=1)

        residual = (weights.unsqueeze(-1) * (history - current.unsqueeze(1))).sum(dim=1)
        fused = current + torch.sigmoid(self.gate_strength) * residual
        return torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)


class SaliencyGatedParallelLayer(ParallelLayer):
    """ParallelLayer extension that recognizes the new depth block only."""

    def __init__(self, input_segments, block_configs, sequential_idx=0):
        # EncoderActorCriticMixin shallow-copies the config before popping its
        # top-level ``class_name``.  Keep this compatibility cleanup local to
        # the experimental class and never mutate the caller's configuration.
        cleaned_configs = deepcopy(block_configs)
        cleaned_configs.pop("class_name", None)
        super().__init__(input_segments, cleaned_configs, sequential_idx)

    def _build_one_block(self, input_segments: Dict[str, tuple], config: dict) -> nn.Module:
        if config.get("class_name") != "SaliencyGatedDepthEncoder":
            return super()._build_one_block(input_segments, config)

        block_config = deepcopy(config)
        component_names = block_config.pop("component_names")
        if len(component_names) != 1:
            raise ValueError("SaliencyGatedDepthEncoder accepts exactly one observation component")
        component_name = component_names[0]
        image_shape = input_segments[component_name]
        output_size = block_config.pop("output_size")
        block_config.pop("class_name")
        block_config.pop("takeout_input_components", None)
        return SaliencyGatedDepthEncoder(image_shape=image_shape, output_size=output_size, **block_config)

    def _run_one_block(self, flat_input, input_segments, input_component_names, block):
        if isinstance(block, SaliencyGatedDepthEncoder):
            input_for_block = get_subobs_by_components(flat_input, input_component_names, input_segments)
            component_shape = input_segments[input_component_names[0]]
            input_for_block = input_for_block.reshape(-1, *component_shape)
            return block(input_for_block)
        return super()._run_one_block(flat_input, input_segments, input_component_names, block)
