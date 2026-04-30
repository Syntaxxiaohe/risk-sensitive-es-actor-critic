# 论文对齐版 Risk-Sensitive Actor-Critic

这个目录是用于复现论文《Risk-sensitive Reinforcement Learning Based on Convex Scoring Functions》的论文对齐版实现。原型版本保留在 `../risk_sensitive_es_ac/`，本目录在原型基础上拆分了 objective、rollout、evaluation、trainer，方便逐步加入更多论文目标和实验。

## 目录结构

- `configs.py`：环境和训练配置
- `envs.py`：标量 OU 统计套利环境
- `objectives.py`：ES/Mean 目标、Bellman target、辅助变量 `v` 的采样和更新
- `networks.py`：Actor/Critic MLP
- `buffers.py`：旧版标量 rollout buffer
- `rollout.py`：旧版标量 rollout，主要用于对照和排错
- `batched_rollout.py`：批量化训练 rollout，可在 CPU/GPU 上一次并行采样多条 episode
- `evaluation.py`：旧版标量 deterministic evaluation 和画图
- `batched_evaluation.py`：批量化 deterministic evaluation，适合大规模 out-of-sample 评估
- `trainer.py`：Actor-Critic 训练主循环
- `main.py`：单次训练 CLI
- `compare.py`：统一 checkpoint 评估 CLI
- `run_multiseed.py`：多 seed、多 objective 实验 runner 和结果汇总

## 环境

建议使用仓库根目录下的 `.venv`：

```powershell
cd D:\math\reinforcementLearning\risk_sensitive_es_ac_paper
..\.venv\Scripts\python -m pip install -r requirements.txt
```

如果要用 Jupyter 交互式理解论文和实验结果：

```powershell
..\.venv\Scripts\python -m pip install -r requirements-notebook.txt
..\.venv\Scripts\python -m jupyter lab
```

Jupyter 内容放在 `notebooks/` 下：

- `01` 到 `05` 是理解型实验 notebook，按环境、实验结果、目标函数、策略可视化、成本分布阅读。
- `notebooks/source_code/` 是源码镜像 notebook，把每个 `.py` 文件拆成可批注的 Jupyter 形式。

如果要使用 GPU，请确认 CUDA 版 PyTorch 可用：

```powershell
..\.venv\Scripts\python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

## 单次训练

默认训练现在使用 `batched` rollout，并且训练中的 validation/final eval 默认使用批量 evaluator。

```powershell
..\.venv\Scripts\python main.py --objective es --device cuda --rollout-mode batched
..\.venv\Scripts\python main.py --objective es06 --device cuda --rollout-mode batched
..\.venv\Scripts\python main.py --objective mean --device cuda --rollout-mode batched
..\.venv\Scripts\python main.py --objective var --device cuda --rollout-mode batched
..\.venv\Scripts\python main.py --objective avar08 --device cuda --rollout-mode batched
..\.venv\Scripts\python main.py --objective mean_var --device cuda --rollout-mode batched
```

如果需要和旧实现对照，可以切回标量 rollout：

```powershell
..\.venv\Scripts\python main.py --objective es --device cpu --rollout-mode scalar --no-vectorized-eval
```

快速 smoke test：

```powershell
..\.venv\Scripts\python main.py --objective es --iterations 2 --num-episodes 8 --eval-episodes 30 --critic-updates 1 --batch-size 16 --output-dir smoke_es --no-heatmap
..\.venv\Scripts\python main.py --objective mean --iterations 2 --num-episodes 8 --eval-episodes 30 --critic-updates 1 --batch-size 16 --output-dir smoke_mean --no-heatmap
```

## Checkpoint 选择

`best_selection` 有两种模式：

- `rollout`：用每轮训练 rollout 的指标选 best，速度快但噪声大
- `validation`：定期跑 deterministic validation 选 best，更稳但更耗时

批量 evaluator 已接入训练 validation，因此现在可以用较大的 validation 设置：

```powershell
..\.venv\Scripts\python main.py --objective es --device cuda --best-selection validation --validation-interval 25 --validation-episodes 5000 --eval-batch-size 65536
```

## 统一比较

训练完成后，可以用 `compare.py` 做正式 out-of-sample 评估。建议正式比较使用 GPU + vectorized evaluator。

```powershell
..\.venv\Scripts\python compare.py --es-dir outputs --mean-dir outputs_mean --eval-episodes 1000000 --vectorized --eval-batch-size 262144 --device cuda --output-dir comparison_outputs
```

`compare.py` 会比较：

- `always_zero`
- `es_last`
- `es_best`
- `mean_last`
- `mean_best`

重点指标：

- `mean_cost`：平均成本，越低越好
- `variance`/`std_cost`：波动大小，越低越稳
- `VaR`：分位数尾部成本
- `ES`：Expected Shortfall，ES 目标重点看这个
- `ES_0.8` / `ES_0.6`：统一口径的两个 ES 指标
- `AVar_0.8`：非对称方差目标的统一评估指标
- `mean_var_utility`：`mean_cost + variance`，对应论文中 `lambda=1` 的 Mean-Var utility

## 多 seed 实验

`run_multiseed.py` 会按 seed 和 objective 自动训练，再做统一比较和汇总。

```powershell
..\.venv\Scripts\python run_multiseed.py --seeds 7,17,31 --output-root multirun --train-device cuda --rollout-mode batched --compare-device cuda --compare-eval-episodes 1000000 --compare-batch-size 262144
```

如果要按论文 Table 2 的静态 scoring-function 模型一起跑：

```powershell
..\.venv\Scripts\python run_multiseed.py --seeds 7,17,31,43,59 --objectives es mean es06 var avar08 mean_var --iterations 1200 --num-episodes 8192 --validation-interval 50 --validation-episodes 10000 --output-root multirun_batched_1200x8192 --train-device cuda --rollout-mode batched --compare-device cuda --compare-eval-episodes 1000000 --compare-batch-size 262144
```

输出结构：

- `multirun/{objective}_seed{seed}`：每个单独训练 run
- `multirun/comparisons/seed{seed}/comparison_table.csv`：单 seed 比较表
- `multirun/all_comparisons.csv`：所有 seed 的明细汇总
- `multirun/summary_by_model.csv`：按模型聚合的均值和标准差

默认情况下，已有完整训练输出和 comparison 表会被跳过，避免覆盖长时间实验结果。如需强制重跑：

```powershell
..\.venv\Scripts\python run_multiseed.py --seeds 7,17,31 --output-root multirun --force
```

如果只想复用已有训练结果并重算 comparison，使用：

```powershell
..\.venv\Scripts\python run_multiseed.py --seeds 7,17,31 --output-root multirun --force-compare
```

## 推荐实验规模

调试链路：

```text
iterations=300
num_episodes=256
seeds=3
```

中等规模：

```text
iterations=1000
num_episodes=512
seeds=3 到 5
validation_interval=25 或 50
validation_episodes=3000 到 5000
compare_eval_episodes=1000000
```

更接近正式复现实验：

```text
iterations=3000
num_episodes=1024
seeds=10
validation_episodes=5000
compare_eval_episodes=3000000 到 5000000
```

## 当前目标

- `es`：静态 `ES_0.8` 目标，使用 convex scoring function，并用样本分位数更新 `v_star`
- `es06`：静态 `ES_0.6` 目标
- `onestep_es08`：递归一步条件 `ES_0.8` baseline，对齐 `RL-OneStepES0.8`
- `onestep_es06`：递归一步条件 `ES_0.6` baseline，对齐 `RL-OneStepES0.6`
- `mean`：风险中性的 RL-Mean baseline
- `var`：方差目标，`v_star` 对应累计成本均值
- `avar08`：`AVar_0.8` 非对称方差目标，`v_star` 对应 0.8-expectile
- `mean_var`：Mean-Var utility，当前使用论文设置 `lambda=1`
- `always_zero`：只用于评估的不交易 baseline

## 实现假设

- 论文没有指定网络深度和激活函数，本实现沿用原型中的 `2 x 64` Tanh MLP。
- Inventory dynamics 使用 `Q_{t+1}=clip(Q_t+a_t, -q_max, q_max)`。
- Mean objective 复用 5 维网络输入，并设置 `v=0, y=0`。
- 批量 rollout 与标量 rollout 统计等价，但不保证 pathwise identical，因为批量路径使用 Torch RNG。
- validation-based best checkpoint 是工程上的稳定性增强，不是额外的论文算法步骤。

## 后续论文对齐任务

- 已加入 OneStepES baseline；后续需要跑 paper-scale 多 seed 对照，因为它的递归风险 target 和静态 scoring-function objective 不同
- 生成 Table 2 风格的最终论文表格
- 增加 Figure 3 风格的 Critic approximation 检查
- 增加 Figure 5 风格的动态 policy 可视化 `(t, y, P, Q)`

## 2026-04-29 OneStepES baseline

新增两个 objective 名称：

- `onestep_es08`：对齐论文里的 `RL-OneStepES0.8`
- `onestep_es06`：对齐论文里的 `RL-OneStepES0.6`

论文第 5 节把 OneStepES 定义为一步条件风险度量的递归目标：

```text
inf_pi rho(cost_0 + rho(cost_1 + rho(... + rho(cost_T))))
```

这里的 `rho` 分别取 `ES_0.8` 或 `ES_0.6`。它和当前 `es/es06` 不同：`es/es06` 是对整条轨迹 total cost 做一次静态 convex-scoring objective，并用 `v_star` 近似 total-cost VaR；`onestep_es08/onestep_es06` 则在 Bellman target 的每一步对 `cost_t + V_{t+1}` 做局部 ES scoring。

当前实现映射：

- OneStepES 不使用 total-cost 的累计状态 `y`，也不使用全局 `v_star`。
- Critic target 固定当前 `state/action`，额外采样 `--onestep-mc-samples` 个下一步价格冲击，得到 `z_t = cost_t + 1_{not done} V(next_state)` 的条件样本。
- 对每条 transition 的条件样本估计 `q_alpha(z_t)`，再用 `q_alpha + mean(relu(z_t - q_alpha)) / (1 - alpha)` 作为一步递归 ES target。
- 评估仍然统一看 out-of-sample total cost 的 `mean_cost`、`ES_0.8`、`ES_0.6`、`variance` 等指标，方便和论文 Table 2 对齐。

单次 smoke / 训练示例：

```powershell
..\.venv\Scripts\python main.py --objective onestep_es08 --device cuda --rollout-mode batched
..\.venv\Scripts\python main.py --objective onestep_es06 --device cuda --rollout-mode batched
```

默认 `--onestep-mc-samples 64`。如果显存或速度压力较大，可以先用 `--onestep-mc-samples 16` 做调试。

多 seed 示例：

```powershell
..\.venv\Scripts\python run_multiseed.py --seeds 7,17,31,43,59 --objectives es es06 onestep_es08 onestep_es06 mean var avar08 mean_var --iterations 1200 --num-episodes 8192 --validation-interval 50 --validation-episodes 10000 --output-root multirun_with_onestep --train-device cuda --rollout-mode batched --compare-device cuda --compare-eval-episodes 1000000 --compare-batch-size 262144
```

也可以只重跑 OneStepES，并和已有 scoring-function 结果比较：

```powershell
..\.venv\Scripts\python run_multiseed.py --seeds 7,17,31,43,59 --objectives onestep_es08 onestep_es06 --iterations 1200 --num-episodes 8192 --validation-interval 50 --validation-episodes 10000 --output-root multirun_onestep_mc64_1200x8192 --train-device cuda --rollout-mode batched --compare-device cuda --compare-eval-episodes 1000000 --compare-batch-size 262144 --onestep-mc-samples 64
```

`compare.py` 也可以直接评估 OneStepES checkpoint：

```powershell
..\.venv\Scripts\python compare.py --onestep-es08-dir path\to\onestep_es08_run --onestep-es06-dir path\to\onestep_es06_run --vectorized --device cuda
```

## 2026-04-30 Composite ES prototype

New objective name:

- `composite_es`: fixed composite total-cost risk objective

Definition:

```text
J_{lambda, alpha}(pi)
  = (1 - lambda) * E[C^pi] + lambda * ES_alpha(C^pi)
```

where `C^pi` is total trajectory cost. `--risk-alpha` controls `alpha`, and
`--risk-lambda` controls `lambda`. Defaults are `risk_alpha=0.8` and
`risk_lambda=0.5`.

Implementation distinction:

- `es` / `es06` optimize a static total-cost ES scoring-function objective and
  use `v_star` as the total-cost VaR auxiliary variable.
- `composite_es` uses the same total-cost ES scoring target, then adds a
  total-cost mean component at weight `1 - lambda`.
- `onestep_es08` / `onestep_es06` are recursive one-step conditional ES
  baselines and are not used by `composite_es`.

Single-run example:

```powershell
..\.venv\Scripts\python main.py --objective composite_es --risk-alpha 0.8 --risk-lambda 0.5 --device cuda --rollout-mode batched
```

Small smoke example:

```powershell
..\.venv\Scripts\python main.py --objective composite_es --risk-alpha 0.8 --risk-lambda 0.5 --iterations 2 --num-episodes 16 --eval-episodes 50 --critic-updates 1 --batch-size 16 --output-dir smoke_composite_es_l05 --no-heatmap
```

Multi-seed example:

```powershell
..\.venv\Scripts\python run_multiseed.py --seeds 7,17,31 --objectives composite_es --risk-alpha 0.8 --risk-lambda 0.5 --output-root multirun_composite_l05 --train-device cuda --rollout-mode batched --compare-device cuda
```

For the first extension prototype in `EXTENSION_PLAN.md`, run a lambda grid:

```powershell
foreach ($l in 0,0.25,0.5,0.75,1) {
  ..\.venv\Scripts\python run_multiseed.py --seeds 7,17,31,43,59 --objectives composite_es --risk-alpha 0.8 --risk-lambda $l --iterations 1200 --num-episodes 8192 --validation-interval 50 --validation-episodes 10000 --output-root multirun_composite_grid --train-device cuda --rollout-mode batched --compare-device cuda --compare-eval-episodes 1000000 --compare-batch-size 262144
}
```

`run_multiseed.py` writes composite directories and comparison summaries with
the lambda in the run label, for example `composite_es_l0p5_seed7` and
`summary_by_model_composite_es_l0p5.csv`, so multiple lambda values can live
under the same output root without overwriting each other.

## 2026-04-30 Conditioned Composite ES prototype

New objective name:

- `conditioned_composite_es`: discrete preference-conditioned composite ES
  actor-critic.

The policy and critic use seven inputs:

```text
(t, v, P, Q, y, lambda, alpha)
```

The first prototype uses a discrete preference grid:

```text
lambda = [0, 0.25, 0.5, 0.75, 1]
alpha = [0.8]
```

Each training iteration samples one grid point from a shuffled cyclic schedule,
runs a full rollout batch at that preference, and updates only that preference's
`v_star_table[(lambda, alpha)]`. The `lambda=0` preference keeps `v=0`, `y=0`,
and uses the mean-style recursive target.

Small smoke example:

```powershell
..\.venv\Scripts\python -B main.py --objective conditioned_composite_es --iterations 2 --num-episodes 8 --eval-episodes 16 --validation-episodes 8 --validation-interval 1 --critic-updates 1 --actor-updates 1 --batch-size 16 --eval-batch-size 16 --conditioned-lambdas 0,0.5,1 --conditioned-alphas 0.8 --conditioned-eval-lambdas 0,0.5,1 --conditioned-calibration-rounds 0 --best-selection validation --output-dir smoke_conditioned_composite_es --device cpu --no-heatmap
```

Prototype-scale example:

```powershell
..\.venv\Scripts\python -B main.py --objective conditioned_composite_es --iterations 1200 --num-episodes 8192 --eval-episodes 1000000 --validation-interval 50 --validation-episodes 10000 --batch-size 512 --conditioned-lambdas 0,0.25,0.5,0.75,1 --conditioned-alphas 0.8 --conditioned-eval-lambdas 0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1 --best-selection validation --device cuda --eval-batch-size 262144 --output-dir conditioned_composite_es_1200x8192 --no-heatmap
```

Important outputs:

- `v_star_table.json`
- `best_v_star_table.json`
- `conditioned_eval_by_theta.csv`
- `comparison_summary.csv`
- `training_composite_metric.png`
