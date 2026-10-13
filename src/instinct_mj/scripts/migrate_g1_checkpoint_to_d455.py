"""Migrate a 29-DoF G1 checkpoint to the 31-DoF D455 policy layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from instinct_mj.assets.unitree_g1 import G1_31DOF_D455_POLICY_JOINT_NAMES


OLD_JOINT_NAMES = (
    "waist_pitch_joint", "waist_roll_joint", "waist_yaw_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
NEW_BODY_NAMES = tuple(G1_31DOF_D455_POLICY_JOINT_NAMES)
NEW_ACTION_NAMES = (*NEW_BODY_NAMES, "head_yaw_joint", "head_pitch_joint")


def _identity(mapping: dict[int, int], old_start: int, new_start: int, size: int) -> None:
    for index in range(size):
        mapping[new_start + index] = old_start + index


def _history(
    mapping: dict[int, int],
    old_start: int,
    new_start: int,
    old_names: tuple[str, ...],
    new_names: tuple[str, ...],
    frames: int,
) -> None:
    old_index = {name: index for index, name in enumerate(old_names)}
    for frame in range(frames):
        for new_joint_index, name in enumerate(new_names):
            if name in old_index:
                mapping[new_start + frame * len(new_names) + new_joint_index] = (
                    old_start + frame * len(old_names) + old_index[name]
                )


def _policy_input_mapping(*, critic: bool) -> dict[int, int]:
    mapping: dict[int, int] = {}
    old_cursor = new_cursor = 0
    if critic:
        _identity(mapping, old_cursor, new_cursor, 24)  # base linear velocity
        old_cursor += 24
        new_cursor += 24
    for _ in range(3):  # angular velocity, gravity, command
        _identity(mapping, old_cursor, new_cursor, 24)
        old_cursor += 24
        new_cursor += 24
    for _ in range(2):  # joint position and velocity histories
        _history(mapping, old_cursor, new_cursor, OLD_JOINT_NAMES, NEW_BODY_NAMES, 8)
        old_cursor += 8 * len(OLD_JOINT_NAMES)
        new_cursor += 8 * len(NEW_BODY_NAMES)
    _history(mapping, old_cursor, new_cursor, OLD_JOINT_NAMES, NEW_ACTION_NAMES, 8)
    old_cursor += 8 * len(OLD_JOINT_NAMES)
    new_cursor += 8 * len(NEW_ACTION_NAMES)
    new_cursor += 8 * 2  # new explicit head-position history: initialize to zero
    _identity(mapping, old_cursor, new_cursor, 128)  # depth latent
    old_cursor += 128
    new_cursor += 128
    expected_old = 920 if critic else 896
    expected_new = 952 if critic else 928
    assert (old_cursor, new_cursor) == (expected_old, expected_new)
    return mapping


def _amp_input_mapping() -> dict[int, int]:
    mapping: dict[int, int] = {}
    _identity(mapping, 0, 0, 30)  # projected gravity
    old_cursor = new_cursor = 30
    for _ in range(2):  # joint position and velocity, ten frames each
        _history(mapping, old_cursor, new_cursor, OLD_JOINT_NAMES, NEW_BODY_NAMES, 10)
        old_cursor += 10 * len(OLD_JOINT_NAMES)
        new_cursor += 10 * len(NEW_BODY_NAMES)
    _identity(mapping, old_cursor, new_cursor, 60)  # base linear/angular velocity
    return mapping


def _remap_columns(old: torch.Tensor, template: torch.Tensor, mapping: dict[int, int]) -> torch.Tensor:
    result = torch.zeros_like(template)
    for new_index, old_index in mapping.items():
        result[:, new_index] = old[:, old_index]
    return result


def migrate(source_path: Path, template_path: Path, output_path: Path) -> None:
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    template = torch.load(template_path, map_location="cpu", weights_only=True)
    old_state = source["model_state_dict"]
    new_state = {key: value.clone() for key, value in template["model_state_dict"].items()}

    # Copy all shape-compatible tensors first. The order-sensitive tensors are
    # overwritten below with explicit semantic mappings.
    for key, value in old_state.items():
        if key in new_state and value.shape == new_state[key].shape:
            new_state[key] = value.clone()

    for prefix, critic in (("actor", False), ("critic", True)):
        mapping = _policy_input_mapping(critic=critic)
        first_layer_keys = [f"{prefix}.gate.0.weight"] + [
            f"{prefix}.experts.{expert}.0.weight" for expert in range(4)
        ]
        for key in first_layer_keys:
            new_state[key] = _remap_columns(old_state[key], new_state[key], mapping)

    old_action_index = {name: index for index, name in enumerate(OLD_JOINT_NAMES)}
    for expert in range(4):
        weight_key = f"actor.experts.{expert}.6.weight"
        bias_key = f"actor.experts.{expert}.6.bias"
        weight = torch.zeros_like(new_state[weight_key])
        bias = torch.zeros_like(new_state[bias_key])
        for new_index, name in enumerate(NEW_ACTION_NAMES):
            if name in old_action_index:
                old_index = old_action_index[name]
                weight[new_index] = old_state[weight_key][old_index]
                bias[new_index] = old_state[bias_key][old_index]
        new_state[weight_key] = weight
        new_state[bias_key] = bias

    std = torch.ones_like(new_state["std"])
    for new_index, name in enumerate(NEW_ACTION_NAMES):
        if name in old_action_index:
            std[new_index] = old_state["std"][old_action_index[name]]
    new_state["std"] = std

    discriminator = {key: value.clone() for key, value in source["discriminator"].items()}
    first_disc_key = "model.model.0.weight"
    discriminator[first_disc_key] = _remap_columns(
        source["discriminator"][first_disc_key],
        template["discriminator"][first_disc_key],
        _amp_input_mapping(),
    )

    output = {
        "model_state_dict": new_state,
        "discriminator": discriminator,
        "iter": 0,
        "infos": {
            "migrated_from": str(source_path),
            "migration": "G1 29-DoF body policy to G1 31-DoF D455; head channels initialized independently",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(f"Saved migrated checkpoint to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    migrate(args.source, args.template, args.output)


if __name__ == "__main__":
    main()
