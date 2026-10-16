# G1 D455 spiral checkpoint — 46550

- Snapshot date: 2026-10-16 (Asia/Shanghai).
- Source: `logs/instinct_rl/g1_d455_parkour/2026-10-16_13-20-30_flat_two_point1m_resume46200/model_46550.pt`.
- Iteration verified from checkpoint: 46550.
- SHA-256: `6a8f0652d0a6d5d85c6a673609b2cc578adafe973017ddfd628f74696da2d48f`.
- Size: 58,466,799 bytes (below GitHub's 100 MiB individual-file limit).
- The run descends from the original `2026-10-14_16-46-03_no_heading_next_platform_resume45500/model_46000.pt`, via 46100, 46150 and 46200 resumes; it is not the older unrelated 46550 checkpoint.
- Task: `Instinct-Parkour-Target-Amp-G1-D455-LegacyHmvTerrain-v0`.
- Terrain proportions: pyramid up/down 20% each, spiral left/right 30% each; retained 180-degree corridors 0%.
- Spiral outer wall 1.6 m, rail 1 m. Recorded reset layer; targets only 1/2 platforms above spawn.
- Rewards: height progress 4, stair evidence 2, openness 0.7, scanning 0.5, wrong-height stairs -3, body follow 1.5.
- Ground flatness uses camera-facing samples at 0 and 1 m. Corridor XY tracking requires mean sole height within strictly 10 cm of target; pyramids always track.
- AMP enabled. GDU visual memory/visual masking not enabled in this run.
- `env.yaml` and `agent.yaml` are copied from the source run; local asset/data paths may require adjustment on another machine.
- Includes policy/optimizer/discriminator states. This is a reproducible snapshot, not a claimed best model or successful evaluation.

Play from the repository root:

```bash
.venv/bin/instinct-play Instinct-Parkour-Target-Amp-G1-D455-LegacyHmvTerrain-v0 --checkpoint-file checkpoints/g1_spiral_flat1m_46550/model_46550.pt --num-envs 16 --device cuda:0 --viewer viser
```
