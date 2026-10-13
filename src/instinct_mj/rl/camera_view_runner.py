"""Checkpoint-compatible runner for the additive 29-to-31 action experiment."""

from __future__ import annotations

import torch
from instinct_rl.runners import OnPolicyRunner


def _copy_or_pad_camera_action_weights(source: dict, target: dict) -> dict:
    """Copy old weights and zero-initialize only the two new camera outputs."""
    adapted = {}
    skipped = []
    for key, target_tensor in target.items():
        source_tensor = source.get(key)
        if source_tensor is None:
            skipped.append(key)
            continue
        if source_tensor.shape == target_tensor.shape:
            adapted[key] = source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)
            continue

        is_actor_output = key.startswith("actor.experts.") and key.rsplit(".", 1)[-1] in {"weight", "bias"}
        if (
            is_actor_output
            and source_tensor.ndim == target_tensor.ndim
            and source_tensor.shape[1:] == target_tensor.shape[1:]
            and target_tensor.shape[0] == source_tensor.shape[0] + 2
        ):
            padded = target_tensor.detach().clone()
            source_on_target = source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)
            padded[: source_tensor.shape[0]] = source_on_target
            padded[source_tensor.shape[0] :] = 0.0
            adapted[key] = padded
            continue

        if key == "std" and source_tensor.ndim == target_tensor.ndim and target_tensor.shape[0] == source_tensor.shape[0] + 2:
            padded = target_tensor.detach().clone()
            source_on_target = source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)
            padded[: source_tensor.shape[0]] = source_on_target
            # Small initial camera exploration; PPO can learn this scale.
            padded[source_tensor.shape[0] :] = 0.05
            adapted[key] = padded
            continue

        raise RuntimeError(
            f"Unexpected checkpoint shape for '{key}': old={tuple(source_tensor.shape)}, "
            f"new={tuple(target_tensor.shape)}"
        )

    if skipped:
        raise RuntimeError(f"checkpoint is missing required model keys: {skipped}")
    return adapted


class CameraViewOnPolicyRunner(OnPolicyRunner):
    """Load the 29-action checkpoint while preserving its original weights."""

    def load(self, path):
        if self.is_mp_rank_other_process():
            return

        loaded = torch.load(path, weights_only=True)
        target_model = self.alg.actor_critic.state_dict()
        adapted_model = _copy_or_pad_camera_action_weights(loaded["model_state_dict"], target_model)
        self.alg.actor_critic.load_state_dict(adapted_model, strict=True)

        # Critic and AMP discriminator dimensions are unchanged.  Reuse their
        # weights but reset the PPO optimizer because two new policy outputs
        # have no old optimizer slots.
        if "discriminator" in loaded:
            self.alg.discriminator.load_state_dict(loaded["discriminator"], strict=True)
        if "discriminator_optimizer" in loaded:
            try:
                self.alg.discriminator_optimizer.load_state_dict(loaded["discriminator_optimizer"])
            except ValueError:
                print("[INFO] Discriminator optimizer state was not compatible; reinitialized it.")

        self.current_learning_iteration = loaded["iter"]
        print(
            "[INFO] Loaded compatible 29-action checkpoint; "
            "initialized 2 camera actions and reset PPO optimizer state."
        )
        return loaded.get("infos")
