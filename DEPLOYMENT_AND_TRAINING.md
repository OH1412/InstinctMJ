# Hiking in the Wild G1 Parkour Deployment

Status: official G1 PPO + AMP full training is running in tmux session `hiking_g1_train`.

## System

- Ubuntu: Ubuntu 22.04.5 LTS
- Kernel: `6.17.0-35-generic` (x86_64)
- GPU: NVIDIA GeForce RTX 5090
- Total VRAM: 32607 MiB
- Driver: 595.71.05
- Driver CUDA compatibility: 13.2
- CPU: 32 logical CPUs
- RAM: 123 GiB total, 97 GiB available at inspection
- Disk: 212 GiB free on `/`
- Python: CPython 3.11.17 in the project uv environment
- uv: 0.12.22

## Repository

- Repository: `project-instinct/InstinctMJ`
- Branch: `main`
- Commit: `6edbc19d4e386ba42e8a0f8c705ec4a2f983ffb`
- Local path: `/home/descfly/humanoid_robot/InstinctMJ`
- MJLab: `/home/descfly/humanoid_robot/mjlab`, tag `v1.6.0`, commit `0fb8a681136be94ffc636a3dd423cabb97d91f10`

## Official Dataset

- Expected archive: `/home/descfly/Downloads/hiking-in-the-wild_Data&Model.zip`
- Archive: 12 MiB, `unzip -tq` passed.
- Extracted directory: `/home/descfly/humanoid_robot/datasets/hiking_in_the_wild`
- AMP dataset directory: `/home/descfly/humanoid_robot/datasets/hiking_in_the_wild/data&model/parkour_motion_reference`
- NPZ: `parkour_motion_without_run_retargetted.npz` (18,982 frames, 29 joints, 50 Hz, 379.64 s)
- YAML: `parkour_motion_without_run.yaml`, selecting the NPZ above
- NPZ keys: `framerate`, `joint_names`, `joint_pos`, `base_pos_w`, `base_quat_w`
- All numeric arrays are finite; root quaternion norms are approximately 1.0.

The Parkour config now points to the extracted directory above. The loader confirmed one motion file and loaded all 18,982 frames; joint mapping is performed by official joint names.

## Environment Setup

Executed from `/home/descfly/humanoid_robot/InstinctMJ`:

```bash
git clone --branch main --single-branch https://github.com/project-instinct/InstinctMJ.git /home/descfly/humanoid_robot/InstinctMJ
git clone --branch v1.6.0 --depth 1 https://github.com/mujocolab/mjlab.git /home/descfly/humanoid_robot/mjlab
uv sync
uv run instinct-list-envs
```

Verified:

- `torch==2.10.0+cu128`, `torch.cuda.is_available() == True`
- CUDA device: NVIDIA GeForce RTX 5090
- `mujoco==3.11.0`, `mujoco-warp==3.11.0`, `warp-lang==1.14.0`
- `mjlab==1.6.0`, `instinct-rl==1.0.2` at the lock-file commit
- Official tasks listed, including `Instinct-Parkour-Target-Amp-G1-v0` and its Play task

## Code Modifications

1. `src/instinct_mj/tasks/parkour/config/parkour_env_cfg.py`
   - Changed `box_height_mean=[0.1, 0.4]` to `(0.1, 0.4)`.
   - Reason: Tyro validates the dataclass default against the declared tuple-or-float type and otherwise refuses to build the official training CLI.
2. `src/instinct_mj/terrains/terrain_generator_cfg.py`
   - Added a local `FlatPatchSamplingCfg` subclass whose `patch_radius` accepts the list/tuple radii used by the Parkour terrain code.
3. `src/instinct_mj/tasks/parkour/config/parkour_env_cfg.py`
   - Parkour terrain config now imports that local compatibility config.
   - Reason: current InstinctMJ terrain code explicitly supports multi-radius foothold sampling, while the locked MJLab base annotation is scalar; without this compatibility type Tyro rejects the official Parkour config.
4. `src/instinct_mj/tasks/parkour/config/g1/g1_parkour_target_amp_cfg.py`
   - Set `_PARKOUR_DATASET_DIR` to the extracted official dataset directory.

No PPO, AMP, depth, MoE, terrain, curriculum, reward, or action module was disabled.

## Environment Validation

- Task listing: passed.
- Config construction: passed.
- Config default env count: 2048.
- Observation groups: `critic`, `policy`, `amp_policy`, `amp_reference`.
- Policy/critic observations include proprioception, commands, previous actions, and `depth_image`.
- AMP groups include projected gravity, relative joint position, joint velocity, base linear/angular velocity.
- Action group: `joint_pos` (official G1 joint-position action interface).
- 1-env launch probe: passed CUDA scene construction, generated 7703 virtual obstacle edges, loaded the AMP motion, and initialized MuJoCo Warp.
- Observed depth shape: `(8, 18, 32)`; actor/critic depth encoders produce 128-D visual features.
- Action shape: 29.
- AMP policy/reference shape: 670 each; discriminator input is 670.

## Smoke Test

Passed with 1 env and 6 iterations. PPO and AMP losses were finite; `model_5.pt` and `model_6.pt` were saved. Resume from `model_5.pt` passed and continued at iteration 5/7 and 6/7.

## GPU Scaling Test

Measured with 3 iterations per candidate:

| num_envs | Peak VRAM | Peak GPU Util | Last Iter Time | Env Steps/s | Result |
|----------|-----------|---------------|----------------|-------------|--------|
| 64       | 1,723 MiB | 54%           | 1.00 s         | 1,538       | pass |
| 256      | 2,901 MiB | 63%           | 1.23 s         | 5,014       | pass |
| 512      | 5,509 MiB | 76%           | 1.33 s         | 9,266       | pass |
| 1024     | 7,297 MiB | 89%           | 1.54 s         | 16,001      | pass |
| 2048     | 13,385 MiB | 76%          | 1.98 s         | 24,785      | pass |
| 4096     | 25,199 MiB | 100%         | 2.99 s         | 32,913      | pass |

Final selected num_envs: `4096`. It had the best measured throughput and approximately 7.2 GiB of VRAM headroom (about 22%).

## Full Training

- Task: `Instinct-Parkour-Target-Amp-G1-v0`
- Command: `uv run instinct-train Instinct-Parkour-Target-Amp-G1-v0 --num-envs 4096 --env.run-name full_4096`
- tmux session: `hiking_g1_train`
- Current PID at last record: `2645210`
- Console log: `/home/descfly/humanoid_robot/InstinctMJ/logs/hiking_g1_train.console.log`
- Run directory: `/home/descfly/humanoid_robot/InstinctMJ/logs/instinct_rl/g1_parkour/2026-10-06_10-58-22_full_4096`
- Last observed: iteration 97/30000, about 32.7k env-steps/s, approximately 3.00 s/iteration, reward about 2.90, discriminator loss 0.287, AMP total loss 0.548; session alive and no OOM.
- Formal checkpoint interval: 5000 iterations.

## Play / Export

The official commands are:

```bash
uv run instinct-play Instinct-Parkour-Target-Amp-G1-Play-v0 --load-run <run_name>
uv run instinct-play Instinct-Parkour-Target-Amp-G1-Play-v0 --load-run <run_name> --export-onnx
```

They require a valid run/checkpoint and the same official dataset configuration.

Smoke checkpoint resume was verified. A parallel play/export attempt using the smoke checkpoint timed out while sharing the GPU with the 4096-env training process and produced no export files; the training session remained healthy. After a formal checkpoint is written, use:

```bash
uv run instinct-play Instinct-Parkour-Target-Amp-G1-Play-v0 \
  --checkpoint-file /absolute/path/to/model_5000.pt \
  --viewer none --max-steps 300

uv run instinct-play Instinct-Parkour-Target-Amp-G1-Play-v0 \
  --checkpoint-file /absolute/path/to/model_5000.pt \
  --viewer none --max-steps 1 --export-onnx True
```

## Monitor / Resume

```bash
tmux attach -t hiking_g1_train
nvidia-smi
tail -f /home/descfly/humanoid_robot/InstinctMJ/logs/hiking_g1_train.console.log
```

To resume a saved formal checkpoint:

```bash
uv run instinct-train Instinct-Parkour-Target-Amp-G1-v0 --num-envs 4096 \
  --agent.resume True \
  --agent.load-run /absolute/path/to/full_4096_run \
  --agent.load-checkpoint model_5000.pt
```

## Blocker and Next Action

Place the original archive at:

`/home/descfly/Downloads/hiking-in-the-wild_Data&Model.zip`

Then rerun the deployment task. The archive must not be replaced by unrelated motion files. Once present, it can be extracted to `/home/descfly/humanoid_robot/datasets/hiking_in_the_wild`, validated against the official AMASS loader, wired into the config, and used for the requested smoke/scaling/full-training sequence.
