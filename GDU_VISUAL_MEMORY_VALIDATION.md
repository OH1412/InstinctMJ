# GDU 视觉记忆代码验证记录

日期：2026-10-15。范围：CPU 代码验证；没有启动新的 MuJoCo 训练或正式记忆训练。

## 使用的真实权重

`logs/instinct_rl/g1_d455_parkour/2026-10-15_13-22-02_height7_stop_at_5cm_resume66000/model_67000.pt`

此权重只用于现有网络兼容性验证，不表示最终 Teacher 已训练完成。严格继承 85 个模型 state_dict 项；原 CNN、Actor Gate/Experts、Critic、std 均包含在内。原 checkpoint 保持原样。

## 执行的验证命令

```bash
.venv/bin/python -m unittest discover -s tests -p test_visual_memory.py -v
.venv/bin/python -m compileall -q src/instinct_mj/rl/visual_memory \
  src/instinct_mj/scripts/visual_memory.py tests/test_visual_memory.py
git diff --check
.venv/bin/python -m instinct_mj.scripts.visual_memory --stage inspect \
  --teacher logs/instinct_rl/g1_d455_parkour/2026-10-15_13-22-02_height7_stop_at_5cm_resume66000/model_67000.pt \
  --layout configs/visual_memory/g1_d455_inspected_layout.json --device cpu
```

最终 unittest 输出：`Ran 29 tests in 3.281s — OK`，无跳过。大部分回归使用 hidden=32 加快 CPU 运行；另有默认 hidden=256 / group=16 的维度验证。没有安装或升级 pytest，使用 Python 自带 unittest。

Student Camera Feedback 增补验证：31 项测试全部通过（2.740 s）；实际 Viser GUI 句柄额外验证了无效时隐藏、恢复显示与原始画面保留。独立 GUI 验证服务器已关闭；未启动 MuJoCo 预览或新的训练。

## 通过的检查

| 检查 | 结果及边界 |
| --- | --- |
| 原权重加载 | 85 个项 strict load，并逐项相等 |
| 初始动作一致性 | 完整有效窗口、同一输入、确定性输出：`rtol=0, atol=0` 通过 |
| GDU 公式 | Eq.18–19 数值校验；没有替换为 GRU |
| 分组预算 | delta 小于、等于、大于 1 三种情况与极端 logits 均校验；组和等于预算 |
| 持续 hidden / Done | 跨步状态变化，只重置 done env |
| Mask yaw / 时间预算 | 进入/退出迟滞、退出提前恢复、任意 episode 前缀 ≤50% |
| 信息隔离 | 无效窗口 CNN 调用次数为 0；改变不可见深度不会改变动作或 hidden |
| 8 帧有效性 | skip=5 / delay=0–1 / refill 验证，失效帧不会提前当作有效窗口 |
| 无视觉状态更新 | 改变本体输入会改变 masked hidden |
| 正常视觉残差 | 投影非零时正常视觉下身体动作受记忆影响，冻结头部动作保持不变 |
| 梯度 | 投影开始学习后 GDU 与 Fusion 正常视觉梯度非零 |
| 恢复 | 视觉恢复沿用此前 hidden |
| PPO | 连续序列、保存的初始状态、Done 对齐；更新前 replay logprob 与 rollout old logprob 一致 |
| PPO 单环境 minibatch | loss 边界展平后 ratio 为 `[T*B]`，不会因原 squeeze 丢失 batch 维而错误广播 |
| 蒸馏 | 固定合成序列上动作 loss 与特征 loss 都下降；不是地形性能实验 |
| 自主反馈 | 合成状态反馈循环中由 Student 动作推进；不是 MuJoCo rollout |
| checkpoint | 保存/恢复/拒绝覆盖/拒绝不匹配架构 |
| 导出 | TorchScript CNN/core，valid 与 masked 条件的 core 输出一致 |
| graph 生命周期 | 连续 100 步 collection hidden 无 grad_fn 累积 |
| AMP | 已安装的真实 WasabiPPO + discriminator 一次合成 update 完成且 loss 有限 |
| 阶段驱动 | distill → dagger → ppo → evaluate 的命令驱动在合成 VecEnv 上执行、落盘和跨阶段加载成功 |
| padding | 无效 padding 步不推进 hidden |
| 历史依赖 | 不同视觉历史、相同 masked 当前输入下身体动作不同；hidden reset 也改变动作 |

测试修复了一个真实兼容问题：已安装 MoeLayer 的二维 einsum 不接受 `[T,B,D]`，新框架在 Teacher/Critic 边界展平并还原时间维，未修改原框架文件。

## 本次没有完成的物理实验

| 指标 | 当前结果 |
| --- | --- |
| 正常视觉地形成功率 | 未运行 |
| 0.2 / 0.5 / 1 / 2 / 3 s 失效成功率 | 未运行 |
| MuJoCo 跌倒率 | 未运行 |
| 真正地形通过性能 | 未运行 |
| Hidden-reset 真实通过率差异 | 未运行 |
| 最终 Teacher / 蒸馏后 Student / best checkpoint | 本次未生成或指定 |

所有测试 checkpoint 和日志都在临时测试目录中，并在测试结束后由 TemporaryDirectory 清理，不是正式模型产物。没有将合成更新称为真实机器人训练成功。

原训练进程 PID `262200` 在最后核对时仍运行；本次未向它发送停止信号，也未替换其 checkpoint 或配置。完整阶段命令与限制见 `GDU_VISUAL_MEMORY_STAGE1.md`。
