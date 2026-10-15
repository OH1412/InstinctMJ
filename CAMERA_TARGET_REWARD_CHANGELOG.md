# Camera Target Reward Changelog

## 2026-10-13 — Live navigation-target gaze

### Problem

The HMV-derived yaw reward classified the depth image as left, forward, or
right and rewarded a fixed camera-to-body yaw (about 45 degrees).  A side state
could remain active on flat terrain, so the camera kept a large relative yaw
after the body had already turned.

### Change

- G1-D455 yaw reward now reads the live world-frame navigation target from
  `PoseVelocityCommand.pos_command_w` every simulation step.
- Desired camera yaw is the current target bearing relative to the robot body,
  clipped only by the physical head-yaw range.
- As the body turns toward the target, desired relative camera yaw continuously
  decreases and reaches zero when the body faces the target.
- Standing and random-velocity terrains have no meaningful position target, so
  their desired camera yaw is zero.
- Body yaw tracking uses the existing live pose-command angular velocity rather
  than a fixed head-angle-derived angular velocity.
- The old fixed-angle side-start penalty and target-view-dependent head/body
  penalty are disabled for G1-D455.
- Pitch behavior is unchanged: it remains terrain-aware (forward on flat
  terrain and downward on rough terrain).

### Training compatibility

The policy observation and action dimensions are unchanged.  Existing D455
checkpoints remain loadable, so training can resume from `model_20000.pt` while
the two head actions adapt to the corrected reward.

### Resumed run

- Source checkpoint:
  `logs/instinct_rl/g1_d455_parkour/2026-10-12_16-54-10_legacy_terrain_seed80000/model_20000.pt`
- New run:
  `logs/instinct_rl/g1_d455_parkour/2026-10-13_12-03-34_live_target_gaze`
- Task: `Instinct-Parkour-Target-Amp-G1-D455-LegacyTerrain-v0`
- Runtime: 2048 environments on `cuda:0`, learning rate `3e-4`
- Validation: the checkpoint loaded without changing network dimensions; a
  one-iteration smoke test completed before the resumed run was started.
