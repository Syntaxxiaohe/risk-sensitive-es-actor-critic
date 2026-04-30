# Risk-Sensitive Actor-Critic with Convex Scoring Functions

This repository contains a paper-aligned implementation of risk-sensitive
actor-critic methods based on convex scoring functions. It includes several
paper-aligned risk objectives, together with two preliminary ES-focused
extensions for composite and preference-conditioned risk-sensitive reinforcement
learning.

The implementation is based on the finite-horizon statistical-arbitrage setting
used in the paper:

> Risk-Sensitive Reinforcement Learning Based on Convex Scoring Functions

The code is intended as a research prototype. It is not a polished library.

## What Is Included

This public repository includes three parts.

First, it includes a paper-aligned reproduction framework for static
total-cost risk-sensitive RL objectives. The implemented objectives include:

- `mean`: risk-neutral mean-cost baseline.
- `es`: static total-cost expected shortfall with alpha = 0.8.
- `es06`: static total-cost expected shortfall with alpha = 0.6.
- `var`: variance objective.
- `avar08`: asymmetric variance objective.
- `mean_var`: mean-variance utility.
- `onestep_es08`: recursive one-step conditional ES baseline with alpha = 0.8.
- `onestep_es06`: recursive one-step conditional ES baseline with alpha = 0.6.

Second, it includes the first extension layer: a fixed composite mean-ES
objective,

```text
J_{lambda, alpha}(pi)
  = (1 - lambda) E[C^pi] + lambda ES_alpha(C^pi),
```

implemented as:

```text
composite_es
```

This objective tests whether a fixed risk-preference parameter can move the
learned policy along an interpretable mean-tail-risk trade-off.

Third, it includes the second extension layer: a discrete
preference-conditioned composite ES actor-critic,

```text
pi_phi(a | s, lambda, alpha),    V_psi(s, lambda, alpha).
```

This is implemented as:

```text
conditioned_composite_es
```

The current prototype fixes `alpha = 0.8` and trains on the discrete lambda
grid:

```text
lambda = [0, 0.25, 0.5, 0.75, 1]
```

It also supports evaluation on held-out lambda values such as:

```text
lambda = [0.125, 0.375, 0.625, 0.875]
```

This repository does not claim to implement continuous lambda/alpha
conditioning. The current conditioned model is a discrete preference-conditioned
prototype.

## Repository Structure

Core implementation:

```text
configs.py                 Environment and training configuration
envs.py                    OU statistical-arbitrage environment
objectives.py              Risk objectives, scoring targets, auxiliary variables
networks.py                Actor and critic networks
rollout.py                 Scalar rollout implementation
batched_rollout.py         Batched rollout implementation
evaluation.py              Scalar evaluation utilities
batched_evaluation.py      Batched evaluation utilities
trainer.py                 Actor-critic training loop
main.py                    Single-run training CLI
compare.py                 Evaluation and comparison CLI
run_multiseed.py           Multi-seed experiment runner
utils.py                   Shared utilities
```

Figure generation:

```text
make_composite_figures.py
make_conditioned_figures.py
```

Included lightweight result figures:

```text
composite_figures/
conditioned_figures/
```

The repository intentionally excludes proposal drafts, personal notes, notebooks,
large training directories, checkpoints, and raw experiment outputs.

## Installation

Create a Python environment and install the minimal dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies are:

```text
numpy
torch
matplotlib
```

CUDA is optional but recommended for larger batched experiments.

## Quick Smoke Test

Run a very small training job:

```bash
python main.py --objective es --iterations 2 --num-episodes 8 --eval-episodes 30 --critic-updates 1 --batch-size 16 --output-dir smoke_es --no-heatmap
```

Run a fixed composite ES smoke test:

```bash
python main.py --objective composite_es --risk-alpha 0.8 --risk-lambda 0.5 --iterations 2 --num-episodes 16 --eval-episodes 50 --critic-updates 1 --batch-size 16 --output-dir smoke_composite_es_l05 --no-heatmap
```

Run a small conditioned composite ES smoke test:

```bash
python main.py --objective conditioned_composite_es --iterations 2 --num-episodes 8 --eval-episodes 16 --validation-episodes 8 --validation-interval 1 --critic-updates 1 --actor-updates 1 --batch-size 16 --eval-batch-size 16 --conditioned-lambdas 0,0.5,1 --conditioned-alphas 0.8 --conditioned-eval-lambdas 0,0.5,1 --conditioned-calibration-rounds 0 --best-selection validation --output-dir smoke_conditioned_composite_es --device cpu --no-heatmap
```

## Example Experiments

Paper-aligned multi-objective run:

```bash
python run_multiseed.py --seeds 7,17,31,43,59 --objectives es mean es06 var avar08 mean_var --iterations 1200 --num-episodes 8192 --validation-interval 50 --validation-episodes 10000 --output-root multirun_batched_1200x8192 --train-device cuda --rollout-mode batched --compare-device cuda --compare-eval-episodes 1000000 --compare-batch-size 262144
```

Fixed composite lambda grid:

```bash
python run_multiseed.py --seeds 7,17,31,43,59 --objectives composite_es --risk-alpha 0.8 --risk-lambda 0.5 --iterations 1200 --num-episodes 8192 --validation-interval 50 --validation-episodes 10000 --output-root multirun_composite_grid --train-device cuda --rollout-mode batched --compare-device cuda --compare-eval-episodes 1000000 --compare-batch-size 262144
```

Discrete preference-conditioned composite ES prototype:

```bash
python main.py --objective conditioned_composite_es --iterations 6000 --num-episodes 8192 --eval-episodes 1000000 --validation-interval 50 --validation-episodes 10000 --batch-size 512 --conditioned-lambdas 0,0.25,0.5,0.75,1 --conditioned-alphas 0.8 --conditioned-eval-lambdas 0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1 --best-selection validation --device cuda --eval-batch-size 262144 --output-dir conditioned_composite_es_6000x8192 --no-heatmap
```

## Preliminary Results

The repository includes lightweight figures and CSV summaries for the composite
and conditioned experiments.

Fixed composite ES results are under:

```text
composite_figures/
```

Conditioned composite ES results are under:

```text
conditioned_figures/
```

These figures summarize:

- fixed composite lambda-grid behavior;
- the mean-cost versus ES0.8 frontier;
- conditioned model behavior on training lambdas;
- conditioned model behavior on held-out lambdas.

The results should be interpreted as preliminary evidence for the proposed
extension, not as a final benchmark.

## Implementation Notes

- The environment is an OU statistical-arbitrage environment with inventory and
  transaction costs.
- The actor and critic use small MLPs.
- Static ES objectives use an auxiliary variable interpreted as a total-cost VaR
  estimate.
- The conditioned prototype maintains preference-specific auxiliary estimates
  for the discrete training grid.
- Evaluation reports out-of-sample total-cost statistics such as mean cost,
  variance, VaR, ES0.8, and ES0.6.

## Current Limitations

- Continuous preference sampling is not implemented.
- Continuous alpha/lambda conditioning is not implemented.
- A distributional critic is not implemented.
- Large training outputs and checkpoints are not included in this public
  repository.

These are natural directions for future work.
