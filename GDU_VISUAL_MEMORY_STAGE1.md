# Hiking 第一阶段：隐式视觉记忆框架

本实现位于 `/home/descfly/humanoid_robot/InstinctMJ`。开发范围是独立代码、CPU 验证、阶段命令和评估框架；没有停止原训练，没有启动新的 MuJoCo 训练，也没有将现有 67000 认定为已完成的最终 Teacher。

## 真实代码检查结果

| 项目 | 实际检查结果 |
| --- | --- |
| 运行框架 | `instinct_rl 1.0.2`；已安装 `rsl-rl-lib 5.4.2`，但当前训练入口使用前者 |
| 仿真依赖 | `mjlab 1.6.0`，MuJoCo / mujoco-warp `3.11.0`，PyTorch `2.10.0` |
| 原策略 | `instinct_rl.modules.all_mixer.EncoderMoEActorCritic` |
| 原 PPO | `instinct_rl.algorithms.wasabi.WasabiPPO`，继承 PPO 并添加 AMP 判别器 |
| 原 Actor 观测 | `5408D`，其中非深度上下文 `800D`，深度 `8×18×32=4608D` |
| 非视觉输入 | 8 步角速度/重力/速度指令历史各 24D，29 路身体关节位置/速度历史各 232D，31 路动作历史 248D，头部位置历史 16D |
| CNN | Conv2d `8→4, k=3, s=1, p=1`、ReLU，flatten `2304`，MLP `2304→256→256→128` |
| Actor MoE | 输入 `800+128=928D`，Gate `928→4`，4 个 Expert 各 `928→256→128→64→31` |
| Critic | 多出特权线速度历史 24D；原始输入 5432D，编码后 952D，4 个 Expert 输出 value |
| 分布 | `Normal(mean, std)`，`std` 是保存于 checkpoint 的 31 维参数 |
| AMP | 启用，判别器输入 670D；头部两关节不在 AMP 关节观测中 |
| Normalizer | 检查的 67000 run `normalizers={}`、`empirical_normalization=false`；加载器也支持显式保存的 Normalizer 并冻结统计 |
| 控制频率 | MuJoCo timestep 0.005 s，decimation 4，策略周期 0.02 s / 50 Hz |
| 深度处理 | crop `(18,0,16,16)`，Gaussian blur `kernel=3, sigma=1`，depth range `[0.4,2.5]` 映射至 `[0,1]` |
| 深度缓存 | `NoisyGroupedRayCasterCamera` 的 37 帧原始采样历史；`delayed_visualizable_image` 输出 8 帧、skip 5、delay 0–1 |
| 深度时间跨度 | 正常控制循环每步 sense 一次，8 帧间隔 0.10 s，首尾跨度 0.70 s；延迟最多另加 0.02 s。不是 8×0.02 s |
| 头部 | `head_yaw_joint` 轴 `(0,0,-1)`，yaw=0 为相对身体正前方；范围约 ±50°。`head_pitch_joint` 轴 `(0,-1,0)`，范围 `[-0.3491,1.57]` |
| 头部控制 | 原策略 31 路动作中的最后两路，经现有 `camera_yaw_pitch` JointPositionActionCfg、原 PD 接口执行 |
| Done/Reset | VecEnv 合并 terminated/truncated；环境自动 reset，返回 reset 后观测；新记忆只对 done 的 env 清零 |
| 原存储 | `RolloutStorage` 保存观测、动作、值、GAE、old mean/std/logprob、actor/critic hidden；其普通 minibatch 会打乱时间 |
| 原循环支持限制 | 标准 Memory 只实现 GRU/LSTM；原 MoE 的 einsum 只支持二维输入，不能直接套用到 GDU 时间维 |
| 原部署 | 现有 CNN/Actor 分离 ONNX 导出没有 GDU hidden 生命周期，新代码提供独立双文件 TorchScript 与显式状态接口 |
| 日志 | 原任务使用 `logs/instinct_rl/g1_d455_parkour/` 下 TensorBoard；新实验使用独立 `logs/visual_memory/` |

以上布局经原日志、当前配置、实际类源码以及 `model_67000.pt` 的权重形状交叉确认。`configs/visual_memory/g1_d455_inspected_layout.json` 仅供 CPU 检查/回归；真实训练始终从环境 observation manager 取得布局，严格加载失败则停止，不做猜测补齐。

## 代码文件

| 文件 | 内容 |
| --- | --- |
| `src/instinct_mj/rl/visual_memory/gdu.py` | `GDUCell`、`GDUMemory`、`MemoryProjection` |
| `src/instinct_mj/rl/visual_memory/policy.py` | `MemoryConfig`、`VisualMemoryPolicy`、`DeploymentCore` |
| `src/instinct_mj/rl/visual_memory/masking.py` | `VisualMaskConfig`、`HeadYawVisualMask`、`DepthValidityHistory` |
| `src/instinct_mj/rl/visual_memory/runtime.py` | `MemoryPerception`、严格 Teacher 加载、独立 checkpoint 保存/恢复 |
| `src/instinct_mj/rl/visual_memory/training.py` | 连续 TBPTT storage、蒸馏损失、`MemoryWasabiPPO`、监督更新 |
| `src/instinct_mj/scripts/visual_memory.py` | inspect/distill/dagger/ppo/evaluate/export 显式阶段命令 |
| `tests/test_visual_memory.py` | 真实权重加载与 CPU 合成序列/驱动链路回归 |
| `.gitignore` | 允许提交这一个新增测试，其他原本忽略的测试内容仍忽略 |

没有修改原 site-packages、原训练脚本、原任务注册、机器人模型、奖励、地形或 PD 控制。没有升级依赖。PPO 的新增子类位于项目源码内，不改变正在运行的 PPO 实例。

## GDU 与融合

采用 [Luo & Yu 原论文](https://arxiv.org/html/1906.08856) Eq. 18–19。默认 hidden=256，16 组，每组 16 维，组预算 delta=1；每组门值总和为 1，每个神经元分配可不同。

```text
d = softmax(group_logits)
delta <= 1: a = delta * d
delta > 1:  a = (M-delta)/(M-1) * d + (delta-1)/(M-1)
candidate = tanh(Ws*x + Us*h_previous + bs)
h = (1-a)*h_previous + a*candidate
```

预算必须在 `(0,M)` 内，可给全部组一个标量，也可逐组指定。没有 GRU reset gate、LSTM cell、显式地图或全局位置输入。

当前真实尺寸：

```text
GDU input = valid visual 128 + nonvisual context 800 + raw head q/dq 4 + validity flags 2
          = 934D
GDU hidden = 256D，两个 affine 输入含旧 hidden 后为 1190D
MemoryProjection = Linear(256,128)，weight/bias 初始化为 0
FusionGate = Linear(1057,128) → ELU → Linear(128,128) → Sigmoid
1057 = visual128 + memory128 + context800 + window_valid1
memory_weight = 1 - window_valid + window_valid * gate
fused_visual = window_valid * valid_visual + memory_weight * memory_feature
原 Actor MoE 输入仍是 800+128=928D
```

正常完整视觉时始终走残差融合路径；`fallback_only` 仅作为 E 消融，正常时将 gate 归零。零投影初始化令完整有效窗口下新旧动作逐元素相等。零投影也意味着第一次 backward 的上游 GDU 梯度为零，投影开始更新后梯度才进入 GDU；测试验证此后正常视觉下的 GDU/Fusion 梯度非零。

Teacher 和 Student 基础网络严格加载全部原参数，包括 Critic 与 std。阶段一只更新 GDU、投影、融合；DAgger/PPO 可显式 `--train-moe-gate`。当前不提供解冻全部专家或 CNN 的默认训练选项。Critic 当前保持冻结，value loss 被记录，但不是阶段一的优化对象。

头部参数另有冻结的原 Actor 分支，记忆损失不经两路 head action 反传。它仍使用原网络/动作/PD 定义；完整视觉时行为等同原策略。失效时它使用 Student 当前可用的视觉特征（无有效 CNN 时为明确的缺失 latent），不访问 Teacher 完整视觉。失效时头部动作无法保证与完整视觉 Teacher 完全一致，也不强制固定头部。全 31 路动作蒸馏中，冻结头部可能产生不可优化的残余误差。

## Mask、缓存和状态

- 正前方 center=0°，进入 ≤15°，退出 ≥20°，使用实际相对身体关节角，不使用全局 heading。
- 默认触发率是每秒 0.3 次的 Poisson hazard，单步概率 `1-exp(-rate*dt)`，不是每步 0.3。
- 持续时间默认 0.2–2.0 s，冷却 0.5 s；duration、rate、yaw 和比例均可配置。
- 采用已累计 episode 时间的预算，任意时刻 sensor mask 时长 ≤已运行时间的 50%。因未知实际终止时间，不以最大 episode 长度预支预算。预算不足时延后触发，单次时长也受已获得预算约束。
- Teacher 原传感器/完整历史保持原样。Student 维护独立的 37 位有效性历史，并按原 observation term 的 delay/skip 选出 8 位。
- Student 整窗只要含无效帧，就将深度输入与有效特征通道隔离，并完全跳过该 env 的 CNN。占位深度不会经 CNN 被解释成障碍；GDU 仍接收本体感知、head q/dq 和有效性。
- sensor 恢复立即开始积累有效帧，CNN 需等待选中的历史全有效，通常最多还需约 0.70–0.72 s；sensor mask ratio 不包含这段额外 memory fallback。
- mask 开始/结束、yaw 离开范围、完整窗口恢复都不清空 hidden。每步更新 GDU；只有 done/reset 或专门的评估消融会清空状态。
- episode 开始时有效性缓存是空的，约 0.70 s 的初始窗口积累会使用 memory fallback；这段不计 sensor mask。Teacher rollout 阶段由 Teacher 控制来保护初期行为。
- 非视觉上下文和当前原始头部状态在 metadata 中追加；metadata 不进入原 MoE，不修改原环境 observation manager。Teacher 标签是另一个 tensor，不是 Student 输入字段。
- 现有 privileged Critic 可读完整视觉；这是独立 value 分支，不把特权视觉或 Teacher 输出送入 Actor/GDU。

## 连续序列与 PPO

默认 TBPTT=64（1.28 s），也支持 128（2.56 s）。保存 pre-action hidden 和 episode-start，保留 rollout 间尚未终止 episode 的状态；优化时只在连续时间上展开，done 位置按元数据重置。采样阶段 hidden 每步 detach，避免 autograd graph 无限积累。

`MemoryWasabiPPO` 复用已安装版本的 clipped PPO、GAE、old logprob、timeout bootstrap、AMP discriminator loss/update。新增存储保留完整时间序列和每段初始状态；该版本要求 rollout horizon=BPTT，并且 env 数可整除 minibatch 数，避免原框架 loss reduction 的分母错误。没有实际 padding，所有步都有效；策略序列 API 仍测试了 padding 时不推进 hidden。

蒸馏使用确定性 Teacher mean 和同一状态下的 Teacher CNN 特征。Feature loss 默认 invalid window 权重 1、normal 0.01、首次恢复窗口 0.25；总动作系数 1、特征系数 0.1。Teacher label 均 stop-gradient。PPO 中这些系数可按 `distill_decay ** iteration` 衰减。

## 启动命令（本次没有执行正式训练）

先确认 Teacher 已训练完成，再设置真实路径。以下 67000 只用于已执行的兼容性测试，不代表选定最终教师。

```bash
cd /home/descfly/humanoid_robot/InstinctMJ
HIKING_TEACHER=/absolute/path/to/completed_teacher/model.pt

# 先在 CPU 检查兼容性；Teacher 旁须有 params/agent.yaml。
.venv/bin/python -m instinct_mj.scripts.visual_memory --stage inspect \
  --teacher "$HIKING_TEACHER" --device cpu \
  --layout configs/visual_memory/g1_d455_inspected_layout.json

# Teacher rollout：先无 mask，再以输出 checkpoint 续接下一难度。
.venv/bin/python -m instinct_mj.scripts.visual_memory --stage distill \
  --teacher "$HIKING_TEACHER" --curriculum-level 0 --num-envs 16 --bptt-length 64

MEMORY_STUDENT=/absolute/path/to/distill/student_1000.pt
.venv/bin/python -m instinct_mj.scripts.visual_memory --stage distill \
  --teacher "$HIKING_TEACHER" --student "$MEMORY_STUDENT" \
  --curriculum-level 1 --num-envs 16 --bptt-length 64

# Student 真正控制机器人，Teacher 仅在相同状态提供标签。
.venv/bin/python -m instinct_mj.scripts.visual_memory --stage dagger \
  --teacher "$HIKING_TEACHER" --student "$MEMORY_STUDENT" \
  --curriculum-level 2 --dagger-teacher-probability 0 --num-envs 16

# 完成 DAgger 后，把 MEMORY_STUDENT 指向其 checkpoint。
.venv/bin/python -m instinct_mj.scripts.visual_memory --stage ppo \
  --teacher "$HIKING_TEACHER" --student "$MEMORY_STUDENT" \
  --curriculum-level 3 --num-envs 16 --bptt-length 128 \
  --action-coef 1 --feature-coef 0.1 --distill-decay 0.999
```

课程 level `0/1/2/3/4/5` 分别为无 mask、最长 `0.2/0.5/1/2/3 s`；阶段由命令固定配置，不自动推进。PPO baseline C 必须显式传 `--allow-untrained-ppo`，避免误将随机记忆置于长时间失效自主控制。混合 DAgger 可用 `--dagger-teacher-probability >0`，日志明确记作 mixed controller，不称为完整 Student rollout。

上述命令复用当前 task factory 的地形/奖励，不会自动将环境还原为 Teacher 当年保存的环境版本。每次独立保存 `env.yaml` 供比对；正式比较必须固定同一份任务代码与配置。若最终 Teacher 改变了观测或动作定义，严格加载会失败，应先核对新布局。

## 评估与消融命令

Viser 增加了独立的 `Student Camera Feedback`：Original 保留实际策略深度帧；Student 在 `Sensor Mask` 或 `CNN History Invalid` 时隐藏图像，全部有效后恢复。两项状态单独标注，随 Viser 选中的机器人切换。原始渲染相机画面也不受 Mask 修改。

```bash
# 仅预览感知失效：未提供 Student checkpoint 时由冻结 Teacher 控制，零优化。
.venv/bin/python -m instinct_mj.scripts.visual_memory --stage preview \
  --teacher "$HIKING_TEACHER" --num-envs 16 --iterations 10000

# 已有训练后的记忆策略时，增加 --student 切换为 Student 控制。
.venv/bin/python -m instinct_mj.scripts.visual_memory --stage preview \
  --teacher "$HIKING_TEACHER" --student "$MEMORY_STUDENT" --num-envs 16
```

普通 `instinct-play` 入口不运行 Mask 调度，请使用上述独立预览入口。访问控制台打印的 Viser 地址；如果现有 8080 已被占用，Viser 会选择其他可用端口。本次只添加代码和验证，未替换正在运行的 Viser 或训练。

```bash
# Baseline A：原模型、完整视觉。
.venv/bin/python -m instinct_mj.scripts.visual_memory --stage evaluate \
  --teacher "$HIKING_TEACHER" --baseline original_full --seed 42

# Baseline B：原模型、无效窗口时用明确缺失的视觉 latent，不伪造深度墙面。
.venv/bin/python -m instinct_mj.scripts.visual_memory --stage evaluate \
  --teacher "$HIKING_TEACHER" --baseline original_masked --seed 42 \
  --mask-min-sec 1 --mask-max-sec 1

# Method F：完整视觉持续残差融合；固定失效时长超过 0.70 s 历史窗口。
.venv/bin/python -m instinct_mj.scripts.visual_memory --stage evaluate \
  --teacher "$HIKING_TEACHER" --student "$MEMORY_STUDENT" --seed 42 \
  --mask-min-sec 1 --mask-max-sec 1

# 相同 checkpoint / seed / mask 规则，在 mask 开始时清空 hidden。
.venv/bin/python -m instinct_mj.scripts.visual_memory --stage evaluate \
  --teacher "$HIKING_TEACHER" --student "$MEMORY_STUDENT" --seed 42 \
  --mask-min-sec 1 --mask-max-sec 1 --reset-memory-on-mask

# 部署导出（输出目录必须不存在），输出 CNN 和带显式 hidden 的 step。
.venv/bin/python -m instinct_mj.scripts.visual_memory --stage export \
  --teacher "$HIKING_TEACHER" --student "$MEMORY_STUDENT" \
  --device cpu --layout configs/visual_memory/g1_d455_inspected_layout.json \
  --output outputs/gdu_export

# CPU 测试：不启动 MuJoCo 仿真或正式训练。
.venv/bin/python -m unittest discover -s tests -p test_visual_memory.py -v
```

对 `0 / 0.2 / 0.5 / 1 / 2 / 3 s` 分别独立运行评估：0 使用 `--no-mask`，其他将 min/max 同设该时长。yaw 与总时间预算仍有效，因此实际实现的 burst 可能少于请求时长，必须核对实际 sensor 时长与 memory fallback；不能把“配置 3 s”直接写成“实际失效 3 s”。

E 消融训练用 `--fusion fallback_only`，F 用默认 `continuous`；恢复训练严格校验此配置，不能将 F checkpoint 静默当成 E 训练。相同地形/种子下真实轨迹一旦动作不同就会分叉；matched-state 历史视觉证据的物理对照实验尚未执行。

## 日志、保存与部署接口

输出为 `logs/visual_memory/{distill,dagger,ppo,evaluate}/时间戳/`。各阶段保存配置、obs schema、Teacher 路径/哈希、Normalizers、模型、optimizer、iteration、随机状态和 PPO 的 AMP 参数；拒绝覆盖已有 checkpoint。

恢复时校验网络/观测 schema，并重新 reset 仿真与 hidden。没有序列化 MuJoCo 物理状态，因此不宣称从保存瞬间逐帧续接物理轨迹。mask/课程配置保存在 config，新的 episode 从干净的 mask 状态开始。

TensorBoard 记录环境 reward、episode length、成功率、fall rate、mask ratio/duration、window valid ratio、memory fallback 累计/连续时长、动作/特征 loss、GDU hidden/gate/group-budget、融合 gate、特征 norm 和梯度 norm；PPO 阶段另有实际 KL、value loss、entropy、AMP losses。成功率按 episode 是否曾同时达到当前 waypoint 的 XY 容差与 5 cm Z 容差定义。`fall_rate` 统计 base_contact/bad_orientation/root_height；另记录包含其他失败终止的 `fall_or_failure_rate`。

评估 JSON 对没有结束的 episode 不给虚构成功率（值为 null）。当前未自动挑选 best model，没有伪造 best 路径；checkpoint 以迭代命名。

部署文件 `depth_encoder.pt` 与 `memory_step.pt`：

```text
hidden = zeros(num_envs,256)
valid window: visual = CNN(depth_window)
invalid window: skip CNN; visual = zeros(num_envs,128)
action,next_hidden = memory_step(context,visual,head_q_dq,
                                 sensor_valid,window_valid,hidden,episode_start)
hidden = next_hidden
done envs: hidden[done] = 0
```

导出 core 不含 Teacher；外部部署端负责复现深度预处理、原 Normalizer（如有）、历史有效性和原 PD/动作接口。没有把旧无状态 ONNX 入口冒充成可直接部署的 GDU 网络。

## 验证范围与未完成实验

真实 67000 checkpoint 的 85 个模型参数/缓冲区均严格继承，正常完整视觉下初始 action 误差为 0。CPU 测试包括真实 WasabiPPO/AMP 合成更新、Teacher rollout → Student DAgger → PPO → evaluate 的合成向量环境驱动，以及不同历史在同一 masked 当前输入下产生不同动作的机制检查。

这些不等于 MuJoCo 崎岖地形成功率。按本次约束，没有正式 Teacher 蒸馏、Student 实机/仿真自主地形 rollout 或 PPO 微调，没有正式 Student checkpoint、最佳模型或性能曲线。正常视觉能力保留、长失效通过率、跌倒率以及 hidden-reset 是否降低真实通过率，均待最终 Teacher 完成后实测。

暂未实现：自动成功率课程、自动 best-model 选择、在线聚合多个历史 DAgger 数据集、全专家/CNN 分阶段解冻、部署整体 ONNX 包、真实 matched-state 地形 A/B 实验。当前是在线 DAgger 式监督（可以混合 controller），所有序列有序 TBPTT；没有将离线随机 MLP minibatch 当作循环学习。

主要后续观察项：早期零投影的梯度启动、Teacher 不可观测的新地形特征误差、初始无历史阶段、冻结头部在长失效时的输出，以及 frozen Critic 对 PPO value 估计的适应性。按实验结果再调整 feature coef、正常视觉辅助系数、预算 delta 和课程长度。
