"""Explicit opt-in GDU stages. Does not launch/stop any existing training.

python -m instinct_mj.scripts.visual_memory --help
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from instinct_mj.rl.visual_memory.masking import VisualMaskConfig
from instinct_mj.rl.visual_memory.policy import DeploymentCore, MemoryConfig, VisualMemoryPolicy
from instinct_mj.rl.visual_memory.runtime import (
    MemoryPerception, load_teacher, restore_memory_checkpoint, save_memory_checkpoint,
)
from instinct_mj.rl.visual_memory.training import MemoryWasabiPPO, supervised_update


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("inspect", "distill", "dagger", "ppo", "evaluate", "export", "preview"), required=True)
    p.add_argument("--teacher", type=Path, required=True, help="Completed pretrained checkpoint; never overwritten")
    p.add_argument("--student", type=Path, help="Stage checkpoint for DAgger/PPO/evaluation/resume")
    p.add_argument("--task", default="Instinct-Parkour-Target-Amp-G1-D455-LegacyHmvTerrain-v0")
    p.add_argument("--layout", type=Path, help="CPU inspect/export only: inspected observation format JSON")
    p.add_argument("--output", type=Path, default=Path("logs/visual_memory"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--save-interval", type=int, default=100)
    p.add_argument("--bptt-length", type=int, choices=(64, 128), default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--groups", type=int, default=16)
    p.add_argument("--budgets", type=float, nargs="+", default=[1.0])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask-max-sec", type=float, default=2.0)
    p.add_argument("--mask-min-sec", type=float, default=0.2)
    p.add_argument("--mask-rate-hz", type=float, default=0.3)
    p.add_argument("--front-center-deg", type=float, default=0.0)
    p.add_argument("--front-enter-deg", type=float, default=15.0)
    p.add_argument("--front-exit-deg", type=float, default=20.0)
    p.add_argument("--mask-cooldown-sec", type=float, default=0.5)
    p.add_argument("--mask-episode-ratio", type=float, default=0.5)
    p.add_argument("--curriculum-level", type=int, choices=range(6))
    p.add_argument("--no-mask", action="store_true")
    p.add_argument("--fusion", choices=("continuous", "fallback_only"), default="continuous")
    p.add_argument("--action-coef", type=float, default=1.0)
    p.add_argument("--feature-coef", type=float, default=0.1)
    p.add_argument("--normal-feature-weight", type=float, default=0.01)
    p.add_argument("--distill-decay", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--dagger-teacher-probability", type=float, default=0.0,
                   help="0 means autonomous Student DAgger; >0 is explicitly mixed rollout")
    p.add_argument("--train-moe-gate", action="store_true")
    p.add_argument("--allow-untrained-ppo", action="store_true", help="Explicit baseline C only")
    p.add_argument("--baseline", choices=("student", "original_full", "original_masked"), default="student")
    p.add_argument("--reset-memory-on-mask", action="store_true")
    return p.parse_args()


def build_environment(args):
    import instinct_mj.tasks  # noqa: F401
    from instinct_mj.tasks.registry import load_env_cfg, load_env_cls, load_vecenv_cls
    cfg = load_env_cfg(args.task, play=False)
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    env = load_env_cls(args.task)(cfg=cfg, device=args.device)
    return load_vecenv_cls(args.task)(env), cfg


def export_artifacts(policy, output):
    output.mkdir(parents=True, exist_ok=False)
    policy.eval()
    device = next(policy.parameters()).device
    example = (torch.zeros(1, policy.context_dim, device=device),
               torch.zeros(1, policy.visual_dim, device=device), torch.zeros(1, 4, device=device),
               torch.ones(1, device=device), torch.ones(1, device=device),
               policy.initial_state(1, device), torch.zeros(1, device=device))
    core = torch.jit.trace(DeploymentCore(policy), example)
    core.save(str(output / "memory_step.pt"))
    depth_encoder = policy.base.encoders._parallel_blocks["depth_encoder"]
    cnn = torch.jit.trace(depth_encoder, torch.zeros(1, *policy.raw_segments["depth_image"], device=device))
    cnn.save(str(output / "depth_encoder.pt"))
    (output / "schema.json").write_text(json.dumps(policy.architecture(), indent=2))
    return core, example


class EpisodeMetrics:
    def __init__(self, n, device):
        self.length = torch.zeros(n, device=device)
        self.had_mask = torch.zeros(n, dtype=torch.bool, device=device)
        self.success = torch.zeros_like(self.had_mask)
        self.completed = self.falls = self.normal_count = self.masked_count = 0
        self.normal_success = self.masked_success = 0
        self.length_sum = 0.0
        self.physical_falls = 0

    def observe(self, env, sensor_mask):
        self.length += 1
        self.had_mask |= sensor_mask
        raw = env.unwrapped
        command = raw.command_manager.get_term("base_velocity")
        robot = raw.scene["robot"]
        foot_ids, _ = robot.find_bodies(("left_ankle_roll_link", "right_ankle_roll_link"), preserve_order=True)
        z = robot.data.body_link_pos_w[:, foot_ids, 2].mean(-1) - 0.035
        xy = (command.pos_command_w[:, :2] - robot.data.root_link_pos_w[:, :2]).norm(dim=-1)
        self.success |= (xy <= command.cfg.target_dis_threshold) & ((z - command.pos_command_w[:, 2]).abs() <= 0.05)

    def finish(self, done, timeouts, physical_falls=None):
        done = done.bool()
        self.completed += int(done.sum())
        self.falls += int((done & ~timeouts.bool()).sum())
        if physical_falls is not None:
            self.physical_falls += int((done & physical_falls.bool()).sum())
        normal, masked = done & ~self.had_mask, done & self.had_mask
        self.normal_count += int(normal.sum())
        self.masked_count += int(masked.sum())
        self.normal_success += int((normal & self.success).sum())
        self.masked_success += int((masked & self.success).sum())
        self.length_sum += float(self.length[done].sum())
        for buf in (self.length, self.had_mask, self.success):
            buf[done] = 0

    def report(self):
        # Non-timeout terminations are a failure proxy (includes out-of-bounds).
        return {"completed_episodes": self.completed,
                "episode_length": self.length_sum / self.completed if self.completed else None,
                "fall_or_failure_rate": self.falls / self.completed if self.completed else None,
                "fall_rate": self.physical_falls / self.completed if self.completed else None,
                "normal_vision_success_rate": self.normal_success / self.normal_count if self.normal_count else None,
                "masked_vision_success_rate": self.masked_success / self.masked_count if self.masked_count else None,
                "terrain_success_rate": (self.normal_success + self.masked_success) / self.completed if self.completed else None,
                "normal_episode_count": self.normal_count, "masked_episode_count": self.masked_count}


def run(args):
    if not args.teacher.is_file():
        raise FileNotFoundError(args.teacher)
    if args.stage in ("dagger", "evaluate") and args.baseline == "student" and not args.student:
        raise ValueError("This stage requires a distilled Student checkpoint")
    if args.stage == "ppo" and not args.student and not args.allow_untrained_ppo:
        raise ValueError("PPO requires distilled Student; baseline C must explicitly opt in")
    if args.stage == "distill" and args.train_moe_gate:
        raise ValueError("Teacher rollout distillation trains only memory/projection/fusion")
    if not 0 <= args.dagger_teacher_probability <= 1:
        raise ValueError("Invalid mixed rollout probability")
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    env = None
    env_cfg = None
    if args.layout:
        if args.stage not in ("inspect", "export"):
            raise ValueError("Training/evaluation requires the real env observation manager")
        obs_format = json.loads(args.layout.read_text())
        obs_format = {g: {k: tuple(v) for k, v in s.items()} for g, s in obs_format.items()}
        checkpoint = torch.load(args.teacher, map_location="cpu", weights_only=True)
        actions = checkpoint["model_state_dict"]["std"].numel()
        num_rewards = 1
    else:
        env, env_cfg = build_environment(args)
        obs_format, actions, num_rewards = env.get_obs_format(), env.num_actions, env.num_rewards
    teacher, normalizers, teacher_data, agent_cfg = load_teacher(args.teacher, obs_format, actions, num_rewards, args.device)
    cfg = MemoryConfig(hidden_dim=args.hidden_dim, num_groups=args.groups,
                       update_budget=args.budgets[0] if len(args.budgets) == 1 else tuple(args.budgets),
                       fusion_mode=args.fusion)
    student = VisualMemoryPolicy(teacher, cfg).to(args.device)
    student.freeze_legacy(train_gate=args.train_moe_gate)
    if args.stage == "inspect":
        print(json.dumps(student.architecture(), indent=2))
        print("Teacher keys loaded strictly:", len(teacher_data["model_state_dict"]))
        if env:
            env.close()
        return
    optimizer = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=args.learning_rate)
    resume_iteration = 0
    if args.student:
        loaded = restore_memory_checkpoint(args.student, student, normalizers=normalizers)
        if loaded["teacher_path"] != str(args.teacher.resolve()):
            raise ValueError("Student checkpoint was trained with a different Teacher path")
        if loaded["config"].get("stage") == args.stage and args.stage != "evaluate":
            optimizer.load_state_dict(loaded["optimizer_state_dict"])
            resume_iteration = loaded["iteration"]
    if args.stage == "export":
        if not args.student:
            raise ValueError("Export requires a Student checkpoint")
        export_artifacts(student, args.output)
        if env:
            env.close()
        return
    assert env is not None
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%f")
    out = args.output / args.stage / stamp
    out.mkdir(parents=True, exist_ok=False)
    writer = SummaryWriter(str(out))
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings.update(memory=asdict(cfg), rollout_controller=("teacher" if args.stage == "distill" or
                    (args.stage == "preview" and args.student is None) else
                    "student" if args.dagger_teacher_probability == 0 else "explicit_mixed"))
    (out / "config.json").write_text(json.dumps(settings, indent=2))
    (out / "obs_format.json").write_text(json.dumps(obs_format, indent=2))
    from mjlab.utils.os import dump_yaml
    from instinct_mj.utils.dict import class_to_dict
    dump_yaml(out / "env.yaml", class_to_dict(env_cfg))
    mask_cfg = VisualMaskConfig(enabled=not args.no_mask, min_duration_s=args.mask_min_sec,
                                max_duration_s=args.mask_max_sec, trigger_rate_hz=args.mask_rate_hz,
                                front_center_deg=args.front_center_deg, enter_deg=args.front_enter_deg,
                                exit_deg=args.front_exit_deg, cooldown_s=args.mask_cooldown_sec,
                                max_episode_ratio=args.mask_episode_ratio)
    if args.curriculum_level is not None:
        levels = (0.0, 0.2, 0.5, 1.0, 2.0, 3.0)
        duration = levels[args.curriculum_level]
        mask_cfg.enabled = duration > 0
        if duration:
            mask_cfg.max_duration_s = duration
            mask_cfg.min_duration_s = min(mask_cfg.min_duration_s, duration)
    settings["visual_mask"] = asdict(mask_cfg)
    if args.baseline == "original_full":
        mask_cfg.enabled = False
    perception = MemoryPerception(env, obs_format, mask_cfg, normalizers, args.seed,
                                  feedback_enabled=args.stage == "preview")
    settings["timing"] = perception.timing()
    (out / "config.json").write_text(json.dumps(settings, indent=2))
    student.initialize_runtime(env.num_envs, args.device)
    if args.stage == "preview":
        from instinct_mj.rl.visual_memory.feedback import run_viser_preview
        try:
            run_viser_preview(env, teacher, student, perception,
                              max_steps=args.iterations * args.bptt_length,
                              teacher_control=args.student is None)
        finally:
            writer.close()
            env.close()
        return
    algorithm = None
    if args.stage == "ppo":
        algo_cfg = copy.deepcopy(agent_cfg["algorithm"])
        algo_cfg.pop("class_name")
        algo_cfg = {k: v for k, v in algo_cfg.items() if v is not None}
        algo_cfg["learning_rate"] = args.learning_rate
        algorithm = MemoryWasabiPPO(student, teacher, bptt_length=args.bptt_length,
                                   action_coef=args.action_coef, feature_coef=args.feature_coef,
                                   normal_feature_weight=args.normal_feature_weight,
                                   distill_decay=args.distill_decay, device=args.device, **algo_cfg)
        memory_format = copy.deepcopy(obs_format)
        memory_format["policy"] = student.obs_segments
        algorithm.init_storage(env.num_envs, args.bptt_length, memory_format, env.num_actions, env.num_rewards)
        algorithm.discriminator.load_state_dict(teacher_data["discriminator"], strict=True)
        if "discriminator_optimizer" in teacher_data:
            algorithm.discriminator_optimizer.load_state_dict(teacher_data["discriminator_optimizer"])
        optimizer = algorithm.optimizer
        if args.student and loaded["config"].get("stage") == "ppo":
            restore_memory_checkpoint(args.student, student, optimizer, normalizers, algorithm)
    obs, info = env.reset()
    perception.reset_done(torch.ones(env.num_envs, dtype=torch.bool, device=env.device))
    metrics = EpisodeMetrics(env.num_envs, env.device)
    prev_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    try:
        for iteration in range(resume_iteration, resume_iteration + args.iterations):
            sequence, full_sequence = [], []
            initial_hidden = student._hidden.detach().clone()
            reward_total = 0.0
            for t in range(args.bptt_length):
                with torch.no_grad():
                    packed, full = perception.prepare(obs)
                    mask = ~perception.sensor_valid
                    if args.reset_memory_on_mask:
                        student.reset(mask & ~prev_mask)
                    prev_mask = mask.clone()
                    # Stored episode-start flags reset states during replay;
                    # the reset-memory ablation is evaluation-only.
                    if args.reset_memory_on_mask and args.stage != "evaluate":
                        raise ValueError("Hidden-reset ablation is evaluation-only")
                    critic = perception.normalize_critic(info["observations"]["critic"])
                    if algorithm:
                        algorithm.storage.teacher_observations[t].copy_(full)
                        action = algorithm.act(packed, critic)
                    else:
                        prediction = student.act_inference(packed)
                        label = teacher.act_inference(full)
                        if args.stage == "distill" or args.baseline == "original_full":
                            action = label
                        elif args.baseline == "original_masked":
                            # Explicit absent visual FEATURE, never fake zero depth.
                            context, z = student.encode_available(packed[:, :student.raw_dim], packed[:, student.raw_dim + 1])
                            action = teacher.actor(torch.cat((context, z), dim=-1))
                        elif args.stage == "dagger" and args.dagger_teacher_probability:
                            choose = torch.rand(env.num_envs, 1, device=env.device) < args.dagger_teacher_probability
                            action = torch.where(choose, label, prediction)
                        else:
                            action = prediction
                    sequence.append(packed.detach())
                    full_sequence.append(full.detach())
                    metrics.observe(env, mask)
                    next_obs, reward, done, next_info = env.step(action)
                    timeouts = next_info.get("time_outs", torch.zeros_like(done))
                    physical_falls = torch.zeros_like(done, dtype=torch.bool)
                    termination_manager = env.unwrapped.termination_manager
                    for reason in ("base_contact", "bad_orientation", "root_height"):
                        if reason in termination_manager.active_terms:
                            physical_falls |= termination_manager.get_term(reason).bool()
                    metrics.finish(done, timeouts, physical_falls)
                    if algorithm:
                        algorithm.process_env_step(reward, done, next_info, next_obs,
                                                   perception.normalize_critic(next_info["observations"]["critic"]))
                    else:
                        student.reset(done)
                    perception.reset_done(done)
                    obs, info = next_obs, next_info
                    reward_total += float(reward.mean())
            if algorithm:
                with torch.no_grad():
                    algorithm.compute_returns(perception.normalize_critic(info["observations"]["critic"]))
                losses, stats = algorithm.update(iteration)
                record = {**losses, **stats}
            elif args.stage in ("distill", "dagger"):
                record = supervised_update(student, teacher, torch.stack(sequence), torch.stack(full_sequence),
                                           initial_hidden, optimizer, args.action_coef, args.feature_coef,
                                           args.normal_feature_weight)
            else:
                record = {}
            record.update(perception.metrics)
            record.update({k: v for k, v in metrics.report().items() if v is not None})
            record["ppo_or_environment_reward"] = reward_total / args.bptt_length
            for k, v in record.items():
                writer.add_scalar(k, float(v), iteration)
            writer.flush()
            if args.stage != "evaluate" and ((iteration + 1) % args.save_interval == 0 or
                                              iteration + 1 == resume_iteration + args.iterations):
                save_memory_checkpoint(out / f"student_{iteration + 1}.pt", student, optimizer,
                                       iteration + 1, settings, args.teacher, normalizers, algorithm)
        (out / "evaluation.json").write_text(json.dumps({**metrics.report(), **perception.metrics,
             "seed": args.seed, "baseline": args.baseline, "reset_memory_on_mask": args.reset_memory_on_mask,
             "teacher": str(args.teacher), "student": str(args.student), "timing": perception.timing(),
             "success_definition": "Ever reached active waypoint XY tolerance and sole Z tolerance 0.05 m",
             "failure_definition": "Non-timeout done; not a pure measured physical fall rate"}, indent=2))
        print(f"Artifacts: {out}")
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    run(parse_args())
