# Parkour Task

## Basic Usage Guidelines

### Unitree H2 (HMV port)

The H2 task includes the physical two-axis head, terrain-only gaze sensors,
HMV gaze rewards, and the left/right stair-corridor terrains:

- `Instinct-Parkour-Target-PPO-H2-v0` (train)
- `Instinct-Parkour-Target-PPO-H2-Play-v0` (play)

```bash
instinct-train Instinct-Parkour-Target-PPO-H2-v0
instinct-play Instinct-Parkour-Target-PPO-H2-Play-v0 --load-run <run_name>
```

The action layout is 29 body joints followed by
`[head_yaw_joint, head_pitch_joint]`. The head limits are yaw
`[-0.8, 0.8]` rad and pitch `[-0.25, 0.5]` rad. Policy observations keep the
two head positions immediately before `depth_image`, so they remain part of
the contiguous proprioceptive input used by ONNX inference.

To create the JetPack 5.1.1 / ONNX Runtime 1.16.3 deployment package added in
HMV commit `555e24d`, run:

```bash
instinct-play Instinct-Parkour-Target-PPO-H2-Play-v0 \
  --load-run <run_name> \
  --export-deployment-package
```

This writes legacy-opset encoder/actor ONNX files under `exported/`, clamps
the ONNX IR version to 9, copies `env.yaml` and `agent.yaml`, and writes a
deployment manifest.

### Parkour Task

**Task IDs:**
- `Instinct-Parkour-Target-Amp-G1-v0` (train)
- `Instinct-Parkour-Target-Amp-G1-Play-v0` (play)

1. Go to `config/g1/g1_parkour_target_amp_cfg.py` and set the `path` and `filtered_motion_selection_filepath` in `AmassMotionCfg` to the reference motion you want to use.

   ```python
   path: str = os.path.expanduser("~/your/path/to/parkour_motion_reference")
   filtered_motion_selection_filepath: str | None = os.path.join(
       path,
       "parkour_motion_without_run.yaml",
   )
   ```

   Keep the selected motion `.npz` files and the selection `.yaml` aligned with the same dataset root unless you intentionally split them.

2. Train the policy:
```bash
instinct-train Instinct-Parkour-Target-Amp-G1-v0
```

3. Play trained policy (`--load-run` must be provided, absolute path is recommended, or use `--agent random` to visualize an untrained policy):

```bash
instinct-play Instinct-Parkour-Target-Amp-G1-Play-v0 --load-run <run_name>
```

4. Export trained policy (`--load-run` must be provided, absolute path is recommended):

```bash
instinct-play Instinct-Parkour-Target-Amp-G1-Play-v0 --load-run <run_name> --export-onnx
```

5. Use the exported ONNX policy for play:

```bash
instinct-play Instinct-Parkour-Target-Amp-G1-Play-v0 --load-run <run_name> --use-onnx
```

## Onboard Joint Order

If you deploy the exported ONNX policy in `instinct_onboard`, make sure `instinct_onboard/instinct_onboard/robot_cfgs.py` uses the `InstinctMJ` MuJoCo joint order instead of the older InstinctLab-style shoulder-first order.

Current `instinct_onboard` `sim_joint_names` order is shoulder-first:

```python
[
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "waist_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "waist_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "waist_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]
```

`InstinctMJ` G1 MJCF order is:

```python
[
    "waist_pitch_joint",
    "waist_roll_joint",
    "waist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
```

When switching `instinct_onboard` to the `InstinctMJ` order, update the following in `instinct_onboard/instinct_onboard/robot_cfgs.py` together:

1. Replace `sim_joint_names` with the `InstinctMJ` order above.
2. Recompute `joint_map` from `real_joint_names` using the same order:

```python
joint_map = [real_joint_names.index(name) for name in sim_joint_names]
```

3. Reorder every array that is documented as "in simulation order" to match the new `sim_joint_names`, especially `joint_signs`, `joint_limits_high`, `joint_limits_low`, and `torque_limits`.

You can safely generate those arrays by name instead of editing indices by hand:

```python
old_sim_joint_names = [
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "waist_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "waist_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "waist_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]

def reorder_from_old(values, sim_joint_names):
    by_name = {name: values[i] for i, name in enumerate(old_sim_joint_names)}
    return [by_name[name] for name in sim_joint_names]
```

This matters because `instinct_onboard/instinct_onboard/agents/base.py` and `instinct_onboard/instinct_onboard/agents/parkour_agent.py` both parse observation and action tensors by iterating `sim_joint_names`. If the onboard order stays on the old InstinctLab convention while the exported `InstinctMJ` config/ONNX uses MJ order, joint observations, action offsets, action scales, and zero-action masks will all be misaligned.

## Common Options

- `--num-envs`: Number of parallel environments (default varies by task)
- `--load-run`: Run name to load checkpoint from for playing
- `--video`: Record training/playback videos
- `--export-onnx`: Export the trained model to ONNX format for onboard deployment during playing
- `--use-onnx`: Use the ONNX model for inference during playing
# Unitree H2

The H2 MuJoCo asset and plain-PPO Parkour tasks are registered as:

```bash
uv run instinct-train Instinct-Parkour-Target-PPO-H2-v0
uv run instinct-play Instinct-Parkour-Target-PPO-H2-Play-v0 --load-run <run_name>
```

The H2 model uses the native 31-joint MuJoCo order, including the two head-gimbal joints.
