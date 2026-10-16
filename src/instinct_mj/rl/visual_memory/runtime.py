"""Read-only perception adapter and strict experiment checkpoint IO."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from instinct_rl.modules import build_actor_critic, build_normalizer

from .masking import DepthValidityHistory, HeadYawVisualMask
from .policy import segments_layout


def load_teacher(checkpoint: Path, obs_format: dict, num_actions: int, num_rewards=1, device="cpu"):
    """Strict loading; future teachers must have a colocated saved agent config."""
    config_path = checkpoint.parent / "params" / "agent.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Teacher agent configuration missing: {config_path}")
    cfg = yaml.safe_load(config_path.read_text())
    policy_cfg = copy.deepcopy(cfg["policy"])
    name = policy_cfg.pop("class_name")
    teacher = build_actor_critic(name, policy_cfg, copy.deepcopy(obs_format), num_actions, num_rewards).to(device)
    data = torch.load(checkpoint, map_location=device, weights_only=True)
    teacher.load_state_dict(data["model_state_dict"], strict=True)
    teacher.eval().requires_grad_(False)
    normalizers = {}
    for group, config in cfg.get("normalizers", {}).items():
        config = dict(config)
        n = build_normalizer(sum(__import__('math').prod(s) for s in obs_format[group].values()),
                             config.pop("class_name"), config).to(device)
        n.load_state_dict(data[f"{group}_normalizer_state_dict"], strict=True)
        normalizers[group] = n.eval().requires_grad_(False)
    unhandled = [k for k in data if k.endswith("normalizer_state_dict") and k.split("_normalizer")[0] not in normalizers]
    if unhandled:
        raise ValueError(f"Cannot silently ignore checkpoint normalizers: {unhandled}")
    return teacher, normalizers, data, cfg


class MemoryPerception:
    """Student stream has its own history validity; original sensor is untouched.

    The complete Teacher tensor is returned separately for labels, never in
    Student forward. The Student depth slice is zeroed if any selected frame
    is invalid; VisualMemoryPolicy then bypasses CNN for that entire row.
    """
    def __init__(self, env, obs_format, mask_cfg, normalizers=None, seed=42, feedback_enabled=False):
        self.env = env
        self.raw = env.unwrapped
        self.layout, self.raw_dim = segments_layout(obs_format["policy"])
        self.depth_shape = tuple(obs_format["policy"]["depth_image"])
        self.normalizers = normalizers or {}
        self.robot = self.raw.scene["robot"]
        self.head_ids, names = self.robot.find_joints(("head_yaw_joint", "head_pitch_joint"), preserve_order=True)
        if list(names) != ["head_yaw_joint", "head_pitch_joint"]:
            raise ValueError("Unexpected head joint ordering")
        term = self.raw.observation_manager.get_term_cfg("policy", "depth_image")
        self.depth_term = term.func
        self.camera = self.raw.scene[term.params["sensor_cfg"].name]
        frames = term.params["num_output_frames"]
        skip = term.params["history_skip_frames"]
        length = self.camera.cfg.data_histories["distance_to_image_plane_noised"]
        self.history = DepthValidityHistory(env.num_envs, length, frames, skip, device=env.device)
        self.mask = HeadYawVisualMask(mask_cfg, env.num_envs, self.raw.step_dt, env.device, seed)
        self.episode_start = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.fallback_duration = torch.zeros(env.num_envs, device=env.device)
        self.fallback_current_duration = torch.zeros_like(self.fallback_duration)
        self.sensor_valid = ~self.mask.mask_active
        self.frame_valid = self.history.sample(torch.zeros(env.num_envs, device=env.device))
        self.last_sensor_frame = self.camera.frame.clone()
        self.metrics = {}
        self.feedback_depth = None
        self.feedback_enabled = feedback_enabled
        self.cnn_window_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset_done(self, done):
        done = done.bool()
        self.mask.reset(done)
        self.history.reset(done)
        self.fallback_duration[done] = 0
        self.fallback_current_duration[done] = 0
        self.episode_start = done.clone()

    def prepare(self, full_obs):
        head = torch.cat((self.robot.data.joint_pos[:, self.head_ids],
                          self.robot.data.joint_vel[:, self.head_ids]), dim=-1).detach()
        # Yaw=0 is straight ahead in the actual MJCF; its axis sign is -Z,
        # which does not change the symmetric front eligibility test.
        self.sensor_valid = self.mask.step(head[:, 0])
        refreshed = (self.camera.frame != self.last_sensor_frame) | self.episode_start
        self.last_sensor_frame.copy_(self.camera.frame)
        self.history.append(self.sensor_valid, refreshed)
        delay = self.depth_term._num_delayed_frames
        self.frame_valid = self.history.sample(delay)
        window_valid = self.frame_valid.all(dim=-1) & self.sensor_valid
        self.cnn_window_valid = window_valid.clone()
        # Read-only viewer copy of the actual policy depth window, before any
        # normalizer or Student masking. This never becomes a Student input.
        if self.feedback_enabled:
            pixels = full_obs[:, self.layout["depth_image"]]
            self.feedback_depth = pixels.reshape(full_obs.shape[0], *self.depth_shape).detach().clone()
        teacher_obs = full_obs.detach().clone()
        if "policy" in self.normalizers:
            teacher_obs = self.normalizers["policy"](teacher_obs)
        student_obs = teacher_obs.clone()
        ds = self.layout["depth_image"]
        student_obs[:, ds] = torch.where(window_valid[:, None], student_obs[:, ds], 0.0)
        meta = torch.cat((self.sensor_valid.float()[:, None], window_valid.float()[:, None],
                          head, self.episode_start.float()[:, None]), dim=-1)
        self.fallback_duration += (~window_valid).float() * self.raw.step_dt
        self.fallback_current_duration = torch.where(window_valid, 0.0,
                                                     self.fallback_current_duration + self.raw.step_dt)
        self.metrics = {
            "sensor_mask_ratio": float(self.mask.mask_active.float().mean()),
            "sensor_mask_total_duration": float(self.mask.mask_total_duration.mean()),
            "sensor_mask_duration": float(self.mask.mask_elapsed_time.mean()),
            "cnn_window_valid_ratio": float(window_valid.float().mean()),
            "memory_fallback_duration": float(self.fallback_duration.mean()),
            "memory_fallback_current_duration": float(self.fallback_current_duration.mean()),
        }
        self.episode_start.zero_()
        return torch.cat((student_obs, meta), dim=-1), teacher_obs

    def normalize_critic(self, observations):
        return self.normalizers["critic"](observations) if "critic" in self.normalizers else observations

    def timing(self):
        return {"control_dt_s": self.raw.step_dt, "frame_stride": self.depth_term.history_skip_frames,
                "history_frames": self.depth_term.num_output_frames,
                "history_span_s": (self.depth_term.num_output_frames - 1) *
                                  self.depth_term.history_skip_frames * self.raw.step_dt,
                "sensor_update_period_s": self.camera.cfg.update_period}


def save_memory_checkpoint(path, student, optimizer, iteration, config, teacher_path,
                           normalizers=None, algorithm=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"schema_version": 1, "architecture": student.architecture(),
            "model_state_dict": student.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration, "config": config, "teacher_path": str(Path(teacher_path).resolve()),
            "teacher_sha256": hashlib.sha256(Path(teacher_path).read_bytes()).hexdigest(),
            "normalizers": {k: v.state_dict() for k, v in (normalizers or {}).items()},
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "runtime_resume": "Fresh env reset and zero hidden; no continuation of simulator state"}
    if algorithm is not None:
        data["amp"] = algorithm.discriminator.state_dict()
        data["amp_optimizer"] = algorithm.discriminator_optimizer.state_dict()
    torch.save(data, path)


def restore_memory_checkpoint(path, student, optimizer=None, normalizers=None, algorithm=None):
    data = torch.load(path, map_location=next(student.parameters()).device, weights_only=True)
    if data["schema_version"] != 1 or data["architecture"] != student.architecture():
        raise ValueError("Memory architecture/observation schema mismatch")
    student.load_state_dict(data["model_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(data["optimizer_state_dict"])
    for key, normalizer in (normalizers or {}).items():
        normalizer.load_state_dict(data["normalizers"][key], strict=True)
    if algorithm is not None and "amp" in data:
        algorithm.discriminator.load_state_dict(data["amp"], strict=True)
        algorithm.discriminator_optimizer.load_state_dict(data["amp_optimizer"])
    torch.set_rng_state(data["torch_rng"].cpu())
    if data.get("cuda_rng") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in data["cuda_rng"]])
    return data
