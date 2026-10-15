# G1 D455 奖励修改与训练权重对照

记录日期：2026-10-15（北京时间）。这是本次核对时的快照，训练仍可能继续产出文件。

## 记录依据与读法

- 训练目录根路径：`logs/instinct_rl/g1_d455_parkour/`，以下 run 名均相对此目录。
- 奖励系数以每个 run 的 `params/env.yaml` 为准；续训来源以 `params/agent.yaml` 的 `load_run`、`load_checkpoint` 为准；产出以实际存在的 `model_*.pt` 为准。
- 奖励函数语义补充自对话记录、现有代码和 `CAMERA_TARGET_REWARD_CHANGELOG.md`。旧函数的精确实现应查对应 run 的 `git/*.diff`；仅比较系数不能发现所有逻辑变化。
- 表中 `500…20000 / 500` 表示从 500 到 20000、间隔 500 的实际保存点。模型编号必须连同 run 目录使用，不能只看编号。
- “无保存权重”不等于没有训练；可能已经进行了更新，但更新没有保存，也没有传递给下次重启。
- `model_66000.pt` 加载到新任务进行播放，只说明用新环境评估旧策略，不表示这个权重学习过新奖励。

## 已保存权重的主要继承链

```text
原始 29-DoF model_80000.pt
  → g1_d455_transfer_seed/model_0.pt（迁移种子，非新增奖励训练产物）
  → 10-12 legacy_terrain_seed80000/model_20000.pt
  → 10-13 live_target_gaze/model_26000.pt
  → legacy_hmv_terrain_continue/model_26500.pt
  → safe_corridor_spawn/model_41500.pt
  → safe_corridor_spawn_resume41500/model_42000.pt
  → latest_terrain_resume42000/model_45500.pt
  → no_heading_next_platform_resume45500/model_46000.pt
  → arm_contact_extra5_resume46000/model_46500.pt
  → pyramid40_corridor60_resume46500/model_66000.pt
  → height7_stop_at_5cm_resume66000/model_66500.pt
```

原始模型路径（来自历史对话）：`logs/instinct_rl/g1_parkour/2026-10-07_12-39-35_full_300k_filtered/model_80000.pt`。迁移种子位于 `logs/instinct_rl/g1_d455_transfer_seed/model_0.pt`。

**10 月 15 日多次修改都从同一个 66000 权重启动，各分支没有依次继承。66500 直接继承 66000，在最后一版配置下训练产生。**

## 正式训练与奖励变化时间线

各行只列主要变化；未列出的系数继承对应阶段配置，完整快照见 run 的 `params/env.yaml`。

| Run | 续训起点 | 奖励/任务变化 | 实际保存权重 |
| --- | --- | --- | --- |
| `2026-10-12_16-54-10_legacy_terrain_seed80000` | 迁移种子 `model_0.pt` | 复用原 80000 的地形；旧式相机 yaw：`camera_yaw_target=1`、侧看启动惩罚 `-10`、`head_body_yaw_l2=-0.1`；`heading_error=-1`、`is_alive=3`、`dont_wait=-0.5`、非脚底接触 `-1` | `500…20000 / 500` |
| `2026-10-13_12-03-34_live_target_gaze` | 上一行 `20000` | 相机目标 yaw 改成实时目标方位；保留 `camera_yaw_target=1`，新增 `camera_yaw_target_error_l2=-2`；移除侧看启动惩罚及 `head_body_yaw_l2`；身体角速度跟踪使用实时导航指令 | `20500…26000 / 500` |
| `2026-10-13_18-10-15_legacy_hmv_terrain_continue` | `live_target_gaze/26000` | 奖励系数不变；融合 HMV 左右走廊 | `26500` |
| `2026-10-13_18-41-46_safe_corridor_spawn` | 上一行 `26500` | 奖励系数不变；走廊安全平台出生及朝向，初始出生点后移（具体地形版本查 run 快照） | `27000…41500 / 500` |
| `2026-10-14_10-41-21_safe_corridor_spawn_resume41500` | 上一行 `41500` | 奖励系数不变，安全出生/地形续训 | `42000、42500`；下一正式分支实际选用 `42000` |
| `2026-10-14_11-56-19_latest_terrain_resume42000` | 上一行 `42000` | 奖励系数不变；使用更新地形。历史修改包含走廊台阶 5–20 cm、扶手/层间空间等，精确几何查本 run 快照 | `42500…45500 / 500` |
| `2026-10-14_16-28-08_ordered_corridor_resume45500` | 上一行 `45500` | 走廊顺序平台目标；奖励系数不变 | 无保存权重 |
| `2026-10-14_16-33-23_next_platform_resume45500` | `latest_terrain_resume42000/45500` | 出生点对应下一个平台目标；奖励系数不变 | 无保存权重 |
| `2026-10-14_16-46-03_no_heading_next_platform_resume45500` | `latest_terrain_resume42000/45500` | 移除 `heading_error=-1`；仍保留 `track_ang_vel_z_exp=1`、`delta_yaw_reward=0.25` | `46000` |
| `2026-10-14_17-30-50_pitch_range_full_flat0773_rough_limit_resume46000` | 上一行 `46000` | pitch 范围 `[-0.3491,1.57]`；平地目标 `0.7730`，崎岖目标 `-0.3491`；pitch 奖励系数仍为 `1 / 1.5` | 无保存权重 |
| `2026-10-14_18-06-43_no_alive_body_contact_resume46000` | `no_heading_next_platform_resume45500/46000` | 移除 `is_alive=3`；`undesired_contacts: -1 → -5`；继承新 pitch 设置 | 无保存权重 |
| `2026-10-14_18-12-18_delayed_stuck_dontwait1_resume46000` | 同一个 `46000` | `dont_wait: -0.5 → -1`；新增 `delayed_stuck=-1`，窗口 1 秒、向目标进度不足 4 cm、指令阈值 0.2 m/s | 无保存权重 |
| `2026-10-14_18-14-40_arm_contact_extra5_resume46000` | 同一个 `46000` | 保留关闭 alive、非脚底 `-5`、dont_wait `-1`、卡死 `-1`；新增 `arm_contacts=-5`，手臂与基础接触项叠加；包含新 pitch 设置 | `46500` |
| `2026-10-14_19-18-56_pyramid40_corridor60_resume46500` | 上一行 `46500` | 奖励不变；正/倒金字塔各 20%，左右走廊各 30%，当前任务仅这四类地形 | `47000…66000 / 500` |
| `2026-10-15_12-32-01_target_height3_no_camera_yaw_resume66000` | 上一行 `66000` | 移除 `camera_yaw_target=1` 及 `camera_yaw_target_error_l2=-2`；新增走廊目标高度奖励 `3`；yaw 平滑惩罚保留 | 无保存权重；TensorBoard 最后记录 `66126` |
| `2026-10-15_12-44-27_active_vision_height_route_resume66000` | 同一个 `66000` | 新增 `active_vision_height_search=2`、`active_vision_body_follow=2`；平地寻找符合目标高低方向的表面，发现后跟随视线 | 无保存权重，未检出训练 scalar |
| `2026-10-15_12-46-35_active_vision_open_height_route_resume66000` | 同一个 `66000` | 搜索加入开阔度偏好 `open_space_scale=0.35`；扫描子系数 `0.25`，总项权重仍 `2`；跟随权重 `2` | 无保存权重；最后记录 `66131` |
| `2026-10-15_13-02-17_active_vision_body_follow5_resume66000` | 同一个 `66000` | 身体跟随 `2 → 5`；搜索保持 `2` | 无保存权重；最后记录 `66066` |
| `2026-10-15_13-11-55_flat_gate_no_target_yaw_resume66000` | 同一个 `66000` | 搜索、开阔度、跟随共用严格平地门；移除 `track_ang_vel_z_exp=1`、`delta_yaw_reward=0.25` | 无保存权重；最后记录 `66006` |
| `2026-10-15_13-14-51_height7_layers2_resume66000` | 同一个 `66000` | 高度奖励 `3 → 7`；出生层相对 `-2/-1/+1/+2` 合法层随机直接目标；加入地面层，过滤越界，episode 内目标固定 | 无保存权重；最后记录 `66045` |
| `2026-10-15_13-22-02_height7_stop_at_5cm_resume66000` | 同一个 `66000` | 高度奖励保持 `7`；脚底平均高度误差 ≤5 cm 后锁定关闭，reset/目标变化重新启用；保留前述全部最终配置 | **`66500`**；核对时最后记录 `66562`，后续产出待补 |

## 当前最重要的权重如何理解

| 权重（必须带 run） | 实际学过的配置 |
| --- | --- |
| `2026-10-14_16-46-03_no_heading_next_platform_resume45500/model_46000.pt` | 实时目标相机 yaw、关闭 heading_error，但 alive 仍为 3、身体朝目标角速度奖励仍存在；尚未学习后续加强接触/卡死/高度/主动视觉配置 |
| `2026-10-14_18-14-40_arm_contact_extra5_resume46000/model_46500.pt` | 关闭 alive、全身非脚底接触 -5、手臂额外 -5、dont_wait -1、延迟卡死 -1、新 pitch 标定；相机和身体仍有目标方向奖励 |
| `2026-10-14_19-18-56_pyramid40_corridor60_resume46500/model_66000.pt` | 继承 46500，并在金字塔 40% / 走廊 60% 下继续训练；**没有学习新目标高度奖励和主动视觉奖励** |
| `2026-10-15_13-22-02_height7_stop_at_5cm_resume66000/model_66500.pt` | 从 66000 在最终配置下继续训练：高度 7 且达标关闭、搜索 2、身体跟随 5、严格平地门、移除直接朝目标转向奖励、出生上下两层内直接目标 |

## 最新阶段的有效奖励系数

以下来自 `2026-10-15_13-22-02_height7_stop_at_5cm_resume66000/params/env.yaml`。

| 奖励项 | 系数 | 说明 |
| --- | ---: | --- |
| `track_lin_vel_xy_exp` | 2.0 | 平面速度跟踪；速度指令由目标 XY 位置生成，不是独立 XY 距离奖励 |
| `dont_wait` | -1.0 | 有前进指令却低速/后退时惩罚 |
| `delayed_stuck` | -1.0 | 延迟判断向当前目标是否缺乏进度 |
| `corridor_target_height` | 7.0 | 走廊脚底平均高度接近目标平台 Z；误差 ≤0.05 m 后锁定关闭 |
| `active_vision_height_search` | 2.0 | 平地搜索、开阔度与符合目标高低方向的表面证据 |
| `active_vision_body_follow` | 5.0 | 平地发现表面后跟随相机记录的世界方向 |
| `undesired_contacts` | -5.0 | 除左右脚底 link 外的身体接触；阈值 1 N |
| `arm_contacts` | -5.0 | 肩/肘/腕/手附加项，与基础项叠加 |
| `camera_yaw_action_acc_l2` | -0.025 | yaw 动作二阶变化惩罚 |
| `camera_pitch_action_acc_l2` | -0.025 | pitch 动作二阶变化惩罚 |
| `flat_forward_camera_pitch` | 1.0 | 平地 pitch 目标 0.7730 rad |
| `rough_downward_camera_pitch` | 1.5 | 崎岖 pitch 目标 -0.3491 rad |
| `stand_still` | -0.3 | 零速度指令时关节运动约束 |
| `volume_points_penetration` | -4.0 | 穿透约束 |
| `feet_air_time` | 0.5 | 脚腾空时间 |
| `feet_slide` | -0.4 | 脚滑动 |
| `joint_deviation_hip` | -0.5 | 髋关节偏离 |
| `ang_vel_xy_l2` | -0.05 | 身体 XY 角速度 |
| `dof_torques_l2` | -1.5e-7 | 关节力矩 |
| `dof_acc_l2` | -1.25e-7 | 关节加速度 |
| `dof_vel_l2` | -1e-4 | 关节速度 |
| `action_rate_l2` | -0.005 | 动作变化 |
| `flat_orientation_l2` | -3.0 | 身体姿态 |
| `pelvis_orientation_l2` | -3.0 | 骨盆姿态 |
| `feet_flat_ori` | -0.4 | 接触时脚姿态 |
| `feet_at_plane` | -0.1 | 脚与地面高度约束 |
| `feet_close_xy` | 0.4 | 脚间距项，实际正负取决于函数输出 |
| `energy` | -5e-5 | 能耗 |
| `freeze_upper_body` | -0.004 | 上肢偏离默认姿态 |
| `dof_pos_limits` | -1.0 | 位置限位 |
| `dof_vel_limits` | -1.0 | 速度限位 |
| `torque_limits` | -0.01 | 力矩限位 |

已移除：`is_alive`、`heading_error`、`track_ang_vel_z_exp`、`delta_yaw_reward`、`camera_yaw_target`、`camera_yaw_target_error_l2`、`camera_yaw_side_start_penalty`、`head_body_yaw_l2`。

主动视觉平地门：走廊存在顺序平台数据，脚下局部区域至少 4 个有效测点、高差 <5 cm，且目标高度差 >10 cm。搜索、开阔度和身体跟随共用此门。目标高度奖励不受此门限制，而是达标后关闭。

身体朝目标的直接角速度奖励已移除，但平面速度指令仍由目标位置生成；不能将当前配置理解为完全取消目标 XY 引导。当前没有独立的“沿崎岖地形延伸方向行走”奖励。

AMP 是独立的判别器奖励来源，不在上述普通环境奖励系数表中；两路头部自由度已从 AMP 关节观测中排除。完整 AMP 超参数见对应 run 的 `params/agent.yaml`。

## 早期实验分支和验证运行

这些分支不在主要继承链中，不能误认为其权重被后续正式训练继承。

| Run | 起点 | 现存 checkpoint |
| --- | --- | --- |
| `2026-10-12_11-04-46` | agent 配置无 resume 来源 | `model_1.pt` |
| `2026-10-12_11-05-31` | agent 配置无 resume 来源 | `500…2500 / 500` |
| `2026-10-12_14-58-11` | 迁移种子 0 | `model_1.pt` |
| `2026-10-12_15-10-02` | 迁移种子 0 | 无 |
| `2026-10-12_15-11-56` | 迁移种子 0 | `model_500.pt、model_1000.pt` |
| `2026-10-12_16-29-31_yaw_gate_smoke` | 迁移种子 0 | `model_1.pt` |
| `2026-10-12_16-30-13_yaw_gate_fixed` | 迁移种子 0 | 无 |
| `2026-10-12_16-34-58_yaw_height_fixed` | 迁移种子 0 | 无 |
| `2026-10-13_12-02-54_live_target_gaze_smoke` | 正式 legacy run 的 20000 | `model_20001.pt`；使用实时目标 yaw 配置 |
| `2026-10-13_18-05-31_legacy_hmv_terrain_smoke` | `live_target_gaze/25500` | 无；正式融合走廊 run 后来从 26000 启动 |

前八个 run 保存的主要奖励系数与旧式相机 yaw 阶段一致；名称中的 gate/fixed 是逻辑或验证差异，不能仅从名字推断新的系数。

## 后续追加记录模板

### 2026-10-15 — 身体持续跟随相机，从最新 67000 续训

- 新 run 后缀：`always_body_follow5_resume67000`；tmux：`g1_d455_always_follow_resume67000`。
- 来源：`2026-10-15_13-22-02_height7_stop_at_5cm_resume66000/model_67000.pt`（实际最新保存点）。
- `active_vision_body_follow=5` 增加 `always_active=True`：所有地形持续根据当前相机相对 yaw 生成身体角速度跟踪目标，不再等待平地发现，不使用历史视线记忆。
- 这覆盖此前“身体跟随仅平地生效”的要求；相机搜索/开阔度/反向高度惩罚仍共用相机前方两点平地门。
- 同时继承当前代码的 XY 奖励在高度奖励完成后启用、反向地形惩罚等修改；这些修改并非源 67000 已学习的内容。
- 本次启动时尚未确认新 checkpoint；持续奖励是软引导，不能保证头身时刻严格对齐。

### 2026-10-15 — 搜索反向高度地形惩罚

- 新 run 后缀 `wrong_height_penalty_resume66000`，来源仍为原 `pyramid40_corridor60_resume46500/model_66000.pt`。
- 新增 `active_vision_wrong_height=-2`：仅平地搜索启用，目标更高却看到低处踏面、或目标更低却看到高处踏面时惩罚；高度阈值 8 cm，证据阈值 0.15。
- 错误方向证据越强，搜索和开阔度奖励越小。错误证据强于正确证据时清除身体跟随记忆，避免跟随错误方向。
- 其余奖励系数不变；当前视野仍可能混合多个表面，此项不是完整楼梯连通性判断。
- 语法与 diff 检查通过；启动时尚未确认本分支保存权重。

### 2026-10-15 — 相机朝向两点平地门

- 续训来源仍为 `2026-10-14_19-18-56_pyramid40_corridor60_resume46500/model_66000.pt`。
- 新分支 run 后缀：`camera_forward_two_point_flat_resume66000`；tmux：`g1_d455_camera_flat2_resume66000`。
- 主动视觉平地门改用独立 `camera_forward_ground` 传感器：相机当前位置地面投影，以及沿相机水平朝向前方 0.25 m，共两条垂直向下射线。
- 两点均有效且地面高度差 <5 cm 才判平坦。删除旧的身体朝向矩形区域/4 点要求；不采后方点。
- 搜索、开阔度、身体跟随共用此门；pitch 原有门控不变。平面速度奖励仍在高度奖励完成后启用。
- 两条射线的局部坐标已验证为 `(0,0,0)` 与 `(0.25,0,0)`。本次启动时未确认新保存权重。

### 2026-10-15 16:10 — 达到目标高度后启用平面速度奖励

- 新 run：`2026-10-15_16-10-01_xy_after_height_resume66000`。
- 实际来源：`2026-10-14_19-18-56_pyramid40_corridor60_resume46500/model_66000.pt`，并非从 66500/67000 继承。
- `track_lin_vel_xy_exp` 系数保持 2，函数改为 `track_lin_vel_xy_after_height_exp`。
- 走廊高度目标未完成时，平面速度奖励为 0；高度误差 ≤5 cm 触发高度奖励锁定关闭后，同一步启用平面速度奖励，并持续至 reset/目标变化。
- 非走廊没有高度奖励，因此平面速度奖励正常启用。
- 高度奖励仍为 7；其他奖励、地形和跨层目标配置沿用上一阶段。
- 启动时无本分支保存权重；训练产出需后续补记。单独验证了达标前后切换、达标后保持启用及非走廊行为。

每次改奖励、门控或目标逻辑都新增记录；不要把上一分支未保存的更新写成继承来源。

```markdown
### YYYY-MM-DD HH:mm — 修改名称

- 新 run：
- 实际加载的源 run / checkpoint：
- 奖励变化（旧系数 → 新系数）：
- 函数、阈值或门控变化：
- 地形比例 / 目标采样变化：
- 已保存 checkpoint（含目录）：
- 最后核对时间 / iteration：
- 状态：初始化 / 训练中 / 已结束 / 无保存权重
- 依据：params/env.yaml、params/agent.yaml、git/*.diff、实际模型文件
```
