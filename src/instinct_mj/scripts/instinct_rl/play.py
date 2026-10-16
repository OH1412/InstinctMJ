"""Play Instinct-RL policies on mjlab tasks."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mjlab
import mujoco
import torch
import tyro
from instinct_rl.runners import OnPolicyRunner
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

import instinct_mj.tasks  # noqa: F401
from instinct_mj.rl import InstinctRlVecEnvWrapper
from instinct_mj.tasks.registry import (
    list_tasks,
    load_env_cfg,
    load_env_cls,
    load_instinct_rl_cfg,
    load_runner_cls,
    load_vecenv_cls,
)


@dataclass(frozen=True)
class PlayConfig:
    agent: Literal["zero", "random", "trained"] = "trained"
    checkpoint_file: str | None = None
    load_run: str | None = None
    checkpoint_pattern: str | None = None
    num_envs: int | None = None
    device: str | None = None
    viewer: Literal["auto", "native", "viser", "none"] = "auto"
    max_steps: int | None = None
    video: bool = False
    video_length: int = 400
    video_dir: str | None = None
    video_height: int | None = None
    video_width: int | None = None
    export_onnx: bool = False
    export_deployment_package: bool = False
    onnx_output_dir: str | None = None
    encoder_opset: int = 12
    actor_opset: int = 12
    max_onnx_ir_version: int = 9
    use_onnx: bool = False
    onnx_model_dir: str | None = None
    no_resume: bool = False
    no_terminations: bool = False
    debug_termination_envs: str | None = None
    """Comma-separated environment indices whose reset reasons are printed."""


class _ViewerEnvAdapter:
    """Adapter so mjlab viewers can consume instinct_rl vec envs."""

    def __init__(self, vec_env: InstinctRlVecEnvWrapper, debug_termination_envs: str | None = None):
        self._vec_env = vec_env
        self.num_envs = vec_env.num_envs
        self._step_count = 0
        self._debug_termination_envs = []
        self._debug_contact_data = None
        model = self.unwrapped.sim.mj_model
        self._goal_site_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"current_goal_{i}")
            for i in range(self.num_envs)
        ]
        if debug_termination_envs:
            self._debug_termination_envs = [int(value.strip()) for value in debug_termination_envs.split(",")]
            invalid = [env_id for env_id in self._debug_termination_envs if not 0 <= env_id < self.num_envs]
            if invalid:
                raise ValueError(f"Termination debug environment indices out of range: {invalid}")
            self._debug_contact_data = mujoco.MjData(self.unwrapped.sim.mj_model)

    @property
    def device(self):
        return self._vec_env.device

    @property
    def cfg(self):
        return self._vec_env.cfg

    @property
    def unwrapped(self):
        return self._vec_env.unwrapped

    def get_observations(self):
        obs, _ = self._vec_env.get_observations()
        self._sync_goal_sites()
        return obs

    def _sync_goal_sites(self):
        if not self._goal_site_ids or min(self._goal_site_ids) < 0:
            return
        env = self.unwrapped
        term = env.command_manager.get_term("base_velocity")
        targets = getattr(term, "pos_command_w", None)
        if targets is None:
            return
        positions = targets.detach().clone()
        positions[:, 2] += 0.06
        # World sites are render-only: update the CPU render model and the
        # current data, without altering collisions, actions or observations.
        env.sim.mj_model.site_pos[self._goal_site_ids] = positions.cpu().numpy()
        env.sim.data.site_xpos[:, self._goal_site_ids] = positions.unsqueeze(0)

    def step(self, actions):
        result = self._vec_env.step(actions)
        self._sync_goal_sites()
        self._step_count += 1
        if self._debug_termination_envs:
            env = self.unwrapped
            for env_id in self._debug_termination_envs:
                if not bool(env.reset_buf[env_id].item()):
                    continue
                reasons = [
                    name
                    for name in env.termination_manager.active_terms
                    if bool(env.termination_manager.get_term(name)[env_id].item())
                ]
                contacts = self._debug_contacts(env_id) if "base_contact" in reasons else []
                print(
                    f"[TERMINATION DEBUG] step={self._step_count} env={env_id} "
                    f"reasons={reasons} contacts={contacts}",
                    flush=True,
                )
        return result

    def _debug_contacts(self, env_id: int) -> list[str]:
        if self._debug_contact_data is None:
            return []
        env = self.unwrapped
        model = env.sim.mj_model
        data = self._debug_contact_data
        sim_data = env.sim.data
        data.qpos[:] = sim_data.qpos[env_id].detach().cpu().numpy()
        data.qvel[:] = sim_data.qvel[env_id].detach().cpu().numpy()
        mujoco.mj_forward(model, data)
        pairs = []
        for contact in data.contact:
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(model.geom_bodyid[geom1])
            body2 = int(model.geom_bodyid[geom2])
            body1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1) or f"body#{body1}"
            body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2) or f"body#{body2}"
            pairs.append(f"{body1_name}<->{body2_name}@{float(contact.dist):.4f}")
        return sorted(set(pairs))

    def reset(self):
        return self._vec_env.reset()

    def close(self):
        return self._vec_env.close()


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _resolve_viewer_backend(viewer: str) -> str:
    if viewer != "auto":
        return viewer
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return "native" if has_display else "viser"


def _disable_headless_debug_visualization(env_cfg, *, keep_command_debug_vis: bool = False) -> None:
    """Disable GUI-dependent debug visualizations for headless play runs."""
    scene = getattr(env_cfg, "scene", None)
    if scene is not None:
        sensors = getattr(scene, "sensors", ())
        for sensor_cfg in sensors:
            if getattr(sensor_cfg, "debug_vis", None) is not None:
                if getattr(sensor_cfg, "reference_entity_name", None) is not None:
                    continue
                sensor_cfg.debug_vis = False

    commands = getattr(env_cfg, "commands", None)
    if isinstance(commands, dict) and not keep_command_debug_vis:
        for command_cfg in commands.values():
            if getattr(command_cfg, "debug_vis", None) is not None:
                command_cfg.debug_vis = False

    observations = getattr(env_cfg, "observations", None)
    if isinstance(observations, dict):
        for group_cfg in observations.values():
            if group_cfg is None:
                continue
            terms = getattr(group_cfg, "terms", None)
            if not isinstance(terms, dict):
                continue
            for term_cfg in terms.values():
                params = getattr(term_cfg, "params", None)
                if isinstance(params, dict) and "debug_vis" in params:
                    params["debug_vis"] = False


def _resolve_rollout_steps(cfg: PlayConfig) -> int | None:
    if cfg.max_steps is not None:
        return cfg.max_steps
    if cfg.viewer == "none":
        return cfg.video_length if cfg.video else 300
    return None


def _force_viewer_realtime_1x(viewer) -> None:
    """Set viewer playback speed to 1x at startup."""
    multipliers = list(getattr(viewer, "SPEED_MULTIPLIERS", ()))
    if len(multipliers) == 0:
        return
    if 1.0 in multipliers:
        speed_index = multipliers.index(1.0)
    else:
        speed_index = min(range(len(multipliers)), key=lambda i: abs(multipliers[i] - 1.0))
    viewer._speed_index = int(speed_index)
    viewer._time_multiplier = float(multipliers[speed_index])


def _patch_world_free_camera(viewer: NativeMujocoViewer) -> None:
    cfg = getattr(viewer, "cfg", None)
    if cfg is None or not hasattr(cfg, "OriginType"):
        return
    if cfg.origin_type != cfg.OriginType.WORLD:
        return
    if not hasattr(viewer, "_setup_camera") or not hasattr(viewer, "_set_camera_world"):
        return

    original_setup_camera = viewer._setup_camera

    def _wrapped_setup_camera() -> None:
        original_setup_camera()
        if getattr(viewer, "viewer", None) is not None:
            viewer._set_camera_world()

    viewer._setup_camera = _wrapped_setup_camera

def _build_dummy_policy(agent_mode: str, action_shape: tuple[int, ...], device: str):
    if agent_mode == "zero":

        def zero_policy(_obs: torch.Tensor) -> torch.Tensor:
            return torch.zeros(action_shape, device=device)

        return zero_policy

    def random_policy(_obs: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.rand(action_shape, device=device) - 1.0

    return random_policy


def _resolve_checkpoint(
    task_id: str,
    cfg: PlayConfig,
    agent_cfg,
) -> Path:
    if cfg.checkpoint_file is not None:
        checkpoint = Path(cfg.checkpoint_file).expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    log_root_path = Path("logs") / "instinct_rl" / agent_cfg.experiment_name
    run_regex = cfg.load_run if cfg.load_run not in (None, "") else agent_cfg.load_run
    if run_regex in (None, ""):
        run_regex = ".*"
    checkpoint_regex = cfg.checkpoint_pattern if cfg.checkpoint_pattern is not None else agent_cfg.load_checkpoint
    if checkpoint_regex in (None, ""):
        checkpoint_regex = "model_.*.pt"

    if not log_root_path.exists():
        raise ValueError(f"Log path does not exist: {log_root_path}")

    candidate_runs: list[Path] = []
    for run in log_root_path.iterdir():
        if not run.is_dir():
            continue
        if run.name == "wandb_checkpoints":
            continue
        # Keep play artifacts out of auto checkpoint lookup by default.
        if run_regex == ".*" and run.name == "_play":
            continue
        if re.match(run_regex, run.name):
            candidate_runs.append(run)

    if len(candidate_runs) == 0:
        raise ValueError(f"No run directories found in {log_root_path} matching '{run_regex}'")
    candidate_runs.sort()

    for run_path in reversed(candidate_runs):
        checkpoints = [
            file.name for file in run_path.iterdir() if file.is_file() and re.match(checkpoint_regex, file.name)
        ]
        if len(checkpoints) == 0:
            continue
        checkpoints.sort(key=lambda name: f"{name:0>15}")
        checkpoint = run_path / checkpoints[-1]
        print(f"[INFO] Auto-selected checkpoint for {task_id}: {checkpoint}")
        return checkpoint

    raise ValueError(
        "No checkpoint file found in matching runs.\n"
        f"  log_path: {log_root_path}\n"
        f"  run_regex: {run_regex}\n"
        f"  checkpoint_regex: {checkpoint_regex}"
    )


def _resolve_video_dir(
    *,
    cfg: PlayConfig,
    task_id: str,
    agent_cfg,
    checkpoint: Path | None,
) -> Path:
    if cfg.video_dir is not None:
        return Path(cfg.video_dir).expanduser().resolve()
    if checkpoint is not None:
        return checkpoint.parent / "videos" / "play"
    return Path("logs") / "instinct_rl" / agent_cfg.experiment_name / task_id / "videos" / "play"


def _resolve_onnx_output_dir(
    *,
    cfg: PlayConfig,
    task_id: str,
    agent_cfg,
    checkpoint: Path | None,
) -> Path:
    if cfg.onnx_output_dir is not None:
        requested = Path(cfg.onnx_output_dir).expanduser().resolve()
        return requested / "exported" if cfg.export_deployment_package else requested
    if cfg.export_deployment_package:
        if checkpoint is None:
            raise ValueError("Deployment-package export requires a checkpoint.")
        return checkpoint.parent / f"deploy_package_{checkpoint.stem}_ir{cfg.max_onnx_ir_version}" / "exported"
    if checkpoint is not None:
        return checkpoint.parent / "exported"
    return Path("logs") / "instinct_rl" / agent_cfg.experiment_name / task_id / "exported"


def _resolve_onnx_model_dir(
    *,
    cfg: PlayConfig,
    agent_cfg,
    checkpoint: Path | None,
) -> Path:
    if cfg.onnx_model_dir is not None:
        return Path(cfg.onnx_model_dir).expanduser().resolve()
    if checkpoint is not None:
        return checkpoint.parent / "exported"
    if cfg.load_run:
        load_run = Path(cfg.load_run).expanduser()
        if load_run.is_absolute():
            return load_run.resolve() / "exported"
        return Path("logs") / "instinct_rl" / agent_cfg.experiment_name / cfg.load_run / "exported"
    run_name = str(getattr(agent_cfg, "run_name", "") or "")
    return Path("logs") / "instinct_rl" / agent_cfg.experiment_name / f"{run_name}_play" / "exported"


def _build_parkour_onnx_policy(
    *,
    vec_env: InstinctRlVecEnvWrapper,
    agent_cfg,
    model_dir: Path,
):
    from instinct_rl.utils.utils import get_subobs_by_components, get_subobs_size

    from instinct_mj.tasks.parkour.scripts.onnxer import load_parkour_onnx_model

    encoder_path = model_dir / "0-depth_encoder.onnx"
    actor_path = model_dir / "actor.onnx"
    missing_files = [str(path) for path in (encoder_path, actor_path) if not path.exists()]
    if missing_files:
        missing_lines = "\n".join(f"  - {path}" for path in missing_files)
        raise FileNotFoundError(f"Missing ONNX model files:\n{missing_lines}")

    obs_segments = vec_env.get_obs_format()["policy"]
    if "depth_image" not in obs_segments:
        raise ValueError("Parkour ONNX inference requires `depth_image` in policy observations.")

    depth_components = agent_cfg.policy.encoder_configs.depth_encoder.component_names

    proprio_components = [
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos",
        "joint_vel",
        "actions",
    ]
    if "camera_yaw_pitch" in obs_segments:
        proprio_components.append("camera_yaw_pitch")
    return load_parkour_onnx_model(
        model_dir=str(model_dir),
        get_subobs_func=lambda obs: get_subobs_by_components(
            obs,
            depth_components,
            obs_segments,
            temporal=True,
        ),
        depth_shape=obs_segments["depth_image"],
        proprio_slice=slice(
            0,
            get_subobs_size(obs_segments, proprio_components),
        ),
    )


def _export_policy_to_onnx(
    *,
    runner,
    vec_env: InstinctRlVecEnvWrapper,
    task_id: str,
    export_dir: Path,
    legacy_jetpack_export: bool = False,
    encoder_opset: int = 12,
    actor_opset: int = 12,
) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    observations, _ = vec_env.get_observations()
    if legacy_jetpack_export:
        from instinct_rl.modules.conv2d import Conv2dHeadModel
        from instinct_rl.modules.transformer import TransformerHeadModel
        from instinct_rl.utils.utils import get_subobs_by_components, module_is_from_type

        actor_critic = runner.alg.actor_critic
        actor_critic.eval()
        encoders = actor_critic.encoders
        exported_paths: list[Path] = []
        with torch.no_grad():
            for block_name, block in encoders._parallel_blocks.items():
                block_config = encoders.block_configs[block_name]
                component_names = block_config["component_names"]
                if isinstance(component_names, str):
                    component_names = [component_names]
                block_input = get_subobs_by_components(
                    observations,
                    component_names,
                    encoders.input_segments,
                    temporal=module_is_from_type(block, TransformerHeadModel),
                )
                if module_is_from_type(block, Conv2dHeadModel):
                    if len(component_names) != 1:
                        raise ValueError(f"Conv2d encoder {block_name} expects one component, got {component_names}")
                    block_input = block_input.reshape(-1, *encoders.input_segments[component_names[0]])
                if module_is_from_type(block, TransformerHeadModel):
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                output_path = export_dir / f"{encoders._sequential_idx}-{block_name}.onnx"
                torch.onnx.export(
                    block,
                    block_input,
                    output_path,
                    input_names=["input"],
                    output_names=["output"],
                    opset_version=encoder_opset,
                    dynamo=False,
                )
                exported_paths.append(output_path)
            encoded_observations = encoders(observations)
            actor_path = export_dir / "actor.onnx"
            torch.onnx.export(
                actor_critic.actor,
                encoded_observations,
                actor_path,
                input_names=["input"],
                output_names=["output"],
                opset_version=actor_opset,
                dynamo=False,
            )
        required_paths = (actor_path, export_dir / "0-depth_encoder.onnx")
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing required exported ONNX files: {missing}; exported={exported_paths}")
    else:
        runner.export_as_onnx(obs=observations, export_model_dir=str(export_dir))

    metadata = {
        "task_id": task_id,
        "obs_format": vec_env.get_obs_format(),
        "num_actions": vec_env.num_actions,
        "num_rewards": vec_env.num_rewards,
    }
    metadata_path = export_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[INFO] Exported ONNX artifacts to: {export_dir}")


def _finalize_deployment_package(
    *,
    export_dir: Path,
    checkpoint: Path,
    task_id: str,
    cfg: PlayConfig,
) -> None:
    """Clamp ONNX IR versions and collect run parameters for JetPack deployment."""
    import onnx

    package_dir = export_dir.parent
    ir_versions: dict[str, int] = {}
    for model_path in sorted(export_dir.glob("*.onnx")):
        model = onnx.load(model_path)
        if model.ir_version > cfg.max_onnx_ir_version:
            model.ir_version = cfg.max_onnx_ir_version
            onnx.save(model, model_path)
        ir_versions[str(model_path.relative_to(package_dir))] = int(onnx.load(model_path).ir_version)

    params_src = checkpoint.parent / "params"
    params_dst = package_dir / "params"
    params_dst.mkdir(parents=True, exist_ok=True)
    for name in ("env.yaml", "agent.yaml"):
        src = params_src / name
        if not src.exists():
            raise FileNotFoundError(f"Missing required run parameter file: {src}")
        shutil.copy2(src, params_dst / name)

    manifest = {
        "task_id": task_id,
        "checkpoint": str(checkpoint.resolve()),
        "target": {
            "jetpack": "5.1.1",
            "python": "3.8",
            "onnxruntime": "1.16.3",
            "arch": "aarch64",
            "max_onnx_ir_version": cfg.max_onnx_ir_version,
        },
        "onnx_opsets": {"encoder": cfg.encoder_opset, "actor": cfg.actor_opset},
        "onnx_ir_versions": ir_versions,
    }
    (package_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[INFO] Deployment package written to: {package_dir}")


def _run_headless_rollout(
    env: _ViewerEnvAdapter,
    policy,
    *,
    num_steps: int | None,
) -> None:
    steps = num_steps if num_steps is not None else 300
    observations = env.get_observations()
    for _ in range(steps):
        with torch.no_grad():
            actions = policy(observations)
            observations, _, _, _ = env.step(actions)


def run_play(task_id: str, cfg: PlayConfig) -> None:
    vecenv_cls = load_vecenv_cls(task_id)
    if vecenv_cls is None:
        raise ImportError(
            "InstinctRlVecEnvWrapper is unavailable. Please install runtime deps:\n"
            '  pip install -e "git+https://github.com/mujocolab/mjlab.git#egg=mjlab"\n'
            '  pip install -e "git+https://github.com/project-instinct/instinct_rl.git#egg=instinct_rl"'
        )
    configure_torch_backends()
    viewer_backend = _resolve_viewer_backend(cfg.viewer)
    # Native viewer should use glfw; headless/video paths should use egl.
    if viewer_backend == "native":
        os.environ["MUJOCO_GL"] = "glfw"
    else:
        os.environ.setdefault("MUJOCO_GL", "egl")

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_instinct_rl_cfg(task_id)
    device = _resolve_device(cfg.device)
    checkpoint: Path | None = None

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if viewer_backend != "native" or not has_display:
        _disable_headless_debug_visualization(env_cfg, keep_command_debug_vis=viewer_backend == "viser")

    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
    if viewer_backend == "viser":
        original_spec_fn = env_cfg.scene.spec_fn
        goal_count = env_cfg.scene.num_envs

        def _add_current_goal_sites(spec):
            if original_spec_fn is not None:
                original_spec_fn(spec)
            for i in range(goal_count):
                spec.worldbody.add_site(
                    name=f"current_goal_{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=(0.18, 0.06, 0.0), rgba=(1.0, 0.1, 0.1, 0.85),
                    group=4, pos=(0.0, 0.0, -100.0),
                )

        env_cfg.scene.spec_fn = _add_current_goal_sites
    if cfg.video_height is not None:
        env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None:
        env_cfg.viewer.width = cfg.video_width
    if cfg.no_terminations:
        env_cfg.terminations = {}
        print("[INFO] All terminations are disabled for play.")

    if cfg.use_onnx and "Parkour" not in task_id:
        raise ValueError("`--use-onnx` currently only supports parkour tasks.")
    if cfg.use_onnx and cfg.agent != "trained":
        raise ValueError("`--use-onnx` only supports `--agent trained`.")

    wants_export = cfg.export_onnx or cfg.export_deployment_package
    if cfg.agent == "trained":
        if cfg.no_resume:
            if wants_export:
                raise ValueError("`--no-resume` cannot be combined with ONNX export.")
            if not cfg.use_onnx:
                raise ValueError("`--no-resume` with `--agent trained` requires `--use-onnx True`.")
            print("[INFO] No-resume mode is enabled. Skip checkpoint loading and use ONNX policy.")
        else:
            checkpoint = _resolve_checkpoint(task_id, cfg, agent_cfg)
    elif wants_export:
        raise ValueError("ONNX export only supports `--agent trained`.")

    env_cls = load_env_cls(task_id)
    env = env_cls(
        cfg=env_cfg,
        device=device,
        render_mode="rgb_array" if cfg.video else None,
    )
    if cfg.video:
        video_dir = _resolve_video_dir(
            cfg=cfg,
            task_id=task_id,
            agent_cfg=agent_cfg,
            checkpoint=checkpoint,
        )
        env = VideoRecorder(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=cfg.video_length,
            disable_logger=True,
            name_prefix=task_id.replace("/", "_"),
        )
        print(f"[INFO] Recording play video to: {video_dir}")

    vec_env = vecenv_cls(
        env,
        policy_group=agent_cfg.policy_observation_group,
        critic_group=agent_cfg.critic_observation_group,
    )

    viewer_env = _ViewerEnvAdapter(vec_env, debug_termination_envs=cfg.debug_termination_envs)

    if cfg.agent in {"zero", "random"}:
        policy = _build_dummy_policy(cfg.agent, (vec_env.num_envs, vec_env.num_actions), device)
    else:
        runner = None
        if checkpoint is not None or wants_export:
            runner_cls = load_runner_cls(task_id) or OnPolicyRunner
            agent_cfg_dict = agent_cfg.to_dict()
            runner = runner_cls(
                vec_env,
                agent_cfg_dict,
                log_dir=None,
                device=device,
            )

        if checkpoint is not None:
            if runner is None:
                raise RuntimeError("Runner is not initialized.")
            runner.load(str(checkpoint))
            print(f"[INFO] Loaded checkpoint: {checkpoint}")

        if wants_export:
            if checkpoint is None:
                raise RuntimeError("checkpoint must be provided when export_onnx is enabled.")
            if runner is None:
                raise RuntimeError("Runner is not initialized.")
            onnx_output_dir = _resolve_onnx_output_dir(
                cfg=cfg,
                task_id=task_id,
                agent_cfg=agent_cfg,
                checkpoint=checkpoint,
            )
            _export_policy_to_onnx(
                runner=runner,
                vec_env=vec_env,
                task_id=task_id,
                export_dir=onnx_output_dir,
                legacy_jetpack_export=cfg.export_deployment_package,
                encoder_opset=cfg.encoder_opset,
                actor_opset=cfg.actor_opset,
            )
            if cfg.export_deployment_package:
                _finalize_deployment_package(
                    export_dir=onnx_output_dir,
                    checkpoint=checkpoint,
                    task_id=task_id,
                    cfg=cfg,
                )

        if cfg.use_onnx:
            onnx_model_dir = _resolve_onnx_model_dir(
                cfg=cfg,
                agent_cfg=agent_cfg,
                checkpoint=checkpoint,
            )
            policy = _build_parkour_onnx_policy(
                vec_env=vec_env,
                agent_cfg=agent_cfg,
                model_dir=onnx_model_dir,
            )
            print(f"[INFO] Loaded ONNX policy from: {onnx_model_dir}")
        else:
            if runner is None:
                raise RuntimeError("Runner is not initialized.")
            policy = runner.get_inference_policy(device=device)

    rollout_steps = _resolve_rollout_steps(cfg)

    if viewer_backend == "native":
        viewer = NativeMujocoViewer(viewer_env, policy)
        _patch_world_free_camera(viewer)
        _force_viewer_realtime_1x(viewer)
        viewer.run(num_steps=rollout_steps)
    elif viewer_backend == "viser":
        viewer = ViserPlayViewer(viewer_env, policy)
        _force_viewer_realtime_1x(viewer)
        viewer.run(num_steps=rollout_steps)
    elif viewer_backend == "none":
        _run_headless_rollout(
            viewer_env,
            policy,
            num_steps=rollout_steps,
        )
    else:
        raise RuntimeError(f"Unsupported viewer backend: {viewer_backend}")

    viewer_env.close()


def main() -> None:
    maybe_print_top_level_help("instinct-play")

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )

    args = tyro.cli(
        PlayConfig,
        args=remaining_args,
        default=PlayConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )
    run_play(chosen_task, args)


if __name__ == "__main__":
    main()
