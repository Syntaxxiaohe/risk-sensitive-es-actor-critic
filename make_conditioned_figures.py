"""Build proposal-ready figures for conditioned composite ES experiments."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


FIXED_TABLE = Path("composite_figures") / "composite_lambda_table.csv"
CONDITIONED_ROOT = Path("multirun_conditioned_composite_6000x8192")
CONDITIONED_SUMMARY = CONDITIONED_ROOT / "summary_conditioned_best_by_lambda.csv"
CONDITIONED_VS_FIXED = CONDITIONED_ROOT / "conditioned_vs_fixed_train_grid.csv"
OUTPUT_DIR = Path("conditioned_figures")


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_fixed_rows() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for row in read_csv(FIXED_TABLE):
        rows.append(
            {
                "lambda": _float(row, "lambda"),
                "mean_cost": _float(row, "mean_cost"),
                "mean_cost_std": _float(row, "mean_cost_std"),
                "ES_0.8": _float(row, "ES_0.8"),
                "ES_0.8_std": _float(row, "ES_0.8_std"),
                "ES_0.6": _float(row, "ES_0.6"),
                "ES_0.6_std": _float(row, "ES_0.6_std"),
                "variance": _float(row, "variance"),
                "variance_std": _float(row, "variance_std"),
            }
        )
    return rows


def read_conditioned_rows() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for row in read_csv(CONDITIONED_SUMMARY):
        rows.append(
            {
                "lambda": _float(row, "lambda"),
                "trained_theta": _int(row, "trained_theta"),
                "num_seeds": _int(row, "num_seeds"),
                "mean_cost": _float(row, "mean_cost_mean"),
                "mean_cost_std": _float(row, "mean_cost_std"),
                "ES_0.8": _float(row, "ES_0_8_mean"),
                "ES_0.8_std": _float(row, "ES_0_8_std"),
                "ES_0.6": _float(row, "ES_0_6_mean"),
                "ES_0.6_std": _float(row, "ES_0_6_std"),
                "variance": _float(row, "variance_mean"),
                "variance_std": _float(row, "variance_std"),
                "composite_metric": _float(row, "composite_metric_mean"),
                "composite_metric_std": _float(row, "composite_metric_std"),
            }
        )
    return rows


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 220,
        }
    )


def save_conditioned_table(conditioned_rows: list[dict[str, float]]) -> None:
    fields = [
        "lambda",
        "trained_theta",
        "num_seeds",
        "mean_cost",
        "mean_cost_std",
        "ES_0.8",
        "ES_0.8_std",
        "ES_0.6",
        "ES_0.6_std",
        "variance",
        "variance_std",
        "composite_metric",
        "composite_metric_std",
    ]
    with (OUTPUT_DIR / "conditioned_lambda_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in conditioned_rows)

    lines = [
        "# Conditioned Composite ES Lambda Grid",
        "",
        "| lambda | theta seen in training? | mean cost | ES0.8 | ES0.6 | variance | composite J |",
        "|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in conditioned_rows:
        trained = "yes" if int(row["trained_theta"]) == 1 else "held-out"
        lines.append(
            f"| {row['lambda']:.3g} | {trained} | {row['mean_cost']:.4f} | "
            f"{row['ES_0.8']:.4f} | {row['ES_0.6']:.4f} | {row['variance']:.4f} | "
            f"{row['composite_metric']:.4f} |"
        )
    (OUTPUT_DIR / "conditioned_lambda_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_train_grid_comparison_table() -> None:
    if not CONDITIONED_VS_FIXED.exists():
        return
    rows = read_csv(CONDITIONED_VS_FIXED)
    lines = [
        "# Conditioned vs Fixed Composite ES on Training Lambda Grid",
        "",
        "| lambda | fixed mean | conditioned mean | fixed ES0.8 | conditioned ES0.8 | fixed variance | conditioned variance |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {_float(row, 'lambda'):.2f} | {_float(row, 'fixed_mean_cost'):.4f} | "
            f"{_float(row, 'conditioned_mean_cost'):.4f} | {_float(row, 'fixed_ES_0_8'):.4f} | "
            f"{_float(row, 'conditioned_ES_0_8'):.4f} | {_float(row, 'fixed_variance'):.4f} | "
            f"{_float(row, 'conditioned_variance'):.4f} |"
        )
    (OUTPUT_DIR / "conditioned_vs_fixed_train_grid.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_metric_comparison(fixed_rows: list[dict[str, float]], conditioned_rows: list[dict[str, float]]) -> None:
    fixed_lambdas = [row["lambda"] for row in fixed_rows]
    cond_lambdas = [row["lambda"] for row in conditioned_rows]
    train_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    trained = [row for row in conditioned_rows if int(row["trained_theta"]) == 1]
    heldout = [row for row in conditioned_rows if int(row["trained_theta"]) == 0]

    metrics = [
        ("mean_cost", "Mean Cost", "#2F6BFF"),
        ("ES_0.8", "ES 0.8", "#D94E41"),
        ("ES_0.6", "ES 0.6", "#D29021"),
        ("variance", "Variance", "#2A9D77"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.8), constrained_layout=True)
    fig.suptitle("Conditioned Composite ES: Fixed Grid vs Preference-Conditioned Model", y=1.02)

    for ax, (key, label, color) in zip(axes.ravel(), metrics):
        fixed_values = [row[key] for row in fixed_rows]
        fixed_stds = [row[f"{key}_std"] for row in fixed_rows]
        cond_values = [row[key] for row in conditioned_rows]
        cond_stds = [row[f"{key}_std"] for row in conditioned_rows]

        ax.errorbar(
            fixed_lambdas,
            fixed_values,
            yerr=fixed_stds,
            color="#555555",
            marker="s",
            linewidth=1.8,
            capsize=3,
            label="separate fixed policies",
        )
        ax.errorbar(
            cond_lambdas,
            cond_values,
            yerr=cond_stds,
            color=color,
            marker="o",
            linewidth=2.0,
            capsize=3,
            label="conditioned policy",
        )
        if heldout:
            ax.scatter(
                [row["lambda"] for row in heldout],
                [row[key] for row in heldout],
                color="white",
                edgecolor=color,
                marker="o",
                s=70,
                linewidth=1.6,
                zorder=5,
                label="held-out lambda",
            )
        ax.set_xlabel("lambda")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        ax.set_xticks(train_ticks)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.savefig(OUTPUT_DIR / "conditioned_vs_fixed_metrics.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "conditioned_vs_fixed_metrics.pdf", bbox_inches="tight")
    plt.close(fig)


def save_frontier(fixed_rows: list[dict[str, float]], conditioned_rows: list[dict[str, float]]) -> None:
    trained = [row for row in conditioned_rows if int(row["trained_theta"]) == 1]
    heldout = [row for row in conditioned_rows if int(row["trained_theta"]) == 0]

    fig, ax = plt.subplots(figsize=(7.3, 5.2), constrained_layout=True)
    ax.errorbar(
        [row["mean_cost"] for row in fixed_rows],
        [row["ES_0.8"] for row in fixed_rows],
        xerr=[row["mean_cost_std"] for row in fixed_rows],
        yerr=[row["ES_0.8_std"] for row in fixed_rows],
        color="#555555",
        marker="s",
        linewidth=1.9,
        capsize=3,
        label="separate fixed policies",
    )
    ax.errorbar(
        [row["mean_cost"] for row in trained],
        [row["ES_0.8"] for row in trained],
        xerr=[row["mean_cost_std"] for row in trained],
        yerr=[row["ES_0.8_std"] for row in trained],
        color="#1F5E7A",
        marker="o",
        linewidth=2.1,
        capsize=3,
        label="conditioned, train lambdas",
    )
    ax.scatter(
        [row["mean_cost"] for row in heldout],
        [row["ES_0.8"] for row in heldout],
        color="white",
        edgecolor="#1F5E7A",
        marker="o",
        s=75,
        linewidth=1.7,
        label="conditioned, held-out lambdas",
        zorder=4,
    )
    for row in fixed_rows:
        ax.annotate(f"{row['lambda']:g}", (row["mean_cost"], row["ES_0.8"]), xytext=(5, 4), textcoords="offset points", fontsize=8, color="#555555")

    ax.set_title("Risk-Return Frontier: Separate Policies vs Conditioned Policy")
    ax.set_xlabel("Mean cost (lower is better)")
    ax.set_ylabel("ES 0.8 total-cost tail risk (lower is better)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.savefig(OUTPUT_DIR / "conditioned_frontier_mean_vs_es08.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "conditioned_frontier_mean_vs_es08.pdf", bbox_inches="tight")
    plt.close(fig)


def save_train_grid_gap(fixed_rows: list[dict[str, float]], conditioned_rows: list[dict[str, float]]) -> None:
    fixed_by_lambda = {row["lambda"]: row for row in fixed_rows}
    train_rows = [row for row in conditioned_rows if int(row["trained_theta"]) == 1]
    lambdas = [row["lambda"] for row in train_rows]
    es_gap = [row["ES_0.8"] - fixed_by_lambda[row["lambda"]]["ES_0.8"] for row in train_rows]
    mean_gap = [row["mean_cost"] - fixed_by_lambda[row["lambda"]]["mean_cost"] for row in train_rows]
    variance_gap = [row["variance"] - fixed_by_lambda[row["lambda"]]["variance"] for row in train_rows]

    fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.plot(lambdas, mean_gap, marker="o", linewidth=2.0, color="#2F6BFF", label="mean cost gap")
    ax.plot(lambdas, es_gap, marker="o", linewidth=2.0, color="#D94E41", label="ES 0.8 gap")
    ax.plot(lambdas, variance_gap, marker="o", linewidth=2.0, color="#2A9D77", label="variance gap")
    ax.set_title("Conditioned Minus Separate Fixed Policy on Training Lambdas")
    ax.set_xlabel("lambda")
    ax.set_ylabel("metric difference")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.savefig(OUTPUT_DIR / "conditioned_minus_fixed_train_grid.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "conditioned_minus_fixed_train_grid.pdf", bbox_inches="tight")
    plt.close(fig)


def save_proposal_text(fixed_rows: list[dict[str, float]], conditioned_rows: list[dict[str, float]]) -> None:
    train_rows = [row for row in conditioned_rows if int(row["trained_theta"]) == 1]
    heldout_rows = [row for row in conditioned_rows if int(row["trained_theta"]) == 0]
    fixed_by_lambda = {row["lambda"]: row for row in fixed_rows}

    train_mean = mean(row["mean_cost"] for row in train_rows)
    train_es = mean(row["ES_0.8"] for row in train_rows)
    train_var = mean(row["variance"] for row in train_rows)
    heldout_mean = mean(row["mean_cost"] for row in heldout_rows)
    heldout_es = mean(row["ES_0.8"] for row in heldout_rows)
    heldout_var = mean(row["variance"] for row in heldout_rows)
    fixed_train_mean = mean(fixed_by_lambda[row["lambda"]]["mean_cost"] for row in train_rows)
    fixed_train_es = mean(fixed_by_lambda[row["lambda"]]["ES_0.8"] for row in train_rows)
    fixed_train_var = mean(fixed_by_lambda[row["lambda"]]["variance"] for row in train_rows)
    es_gap = train_es - fixed_train_es
    var_gap = train_var - fixed_train_var
    mean_gap = train_mean - fixed_train_mean

    best_es_row = min(conditioned_rows, key=lambda row: row["ES_0.8"])
    worst_es_row = max(conditioned_rows, key=lambda row: row["ES_0.8"])

    text = f"""# Proposal Figure/Table Draft: Preference-Conditioned Composite ES

Suggested caption:

Figure X compares separately trained fixed composite policies with a single
preference-conditioned actor-critic trained on the same composite ES family.
The conditioned policy takes `(lambda, alpha)` as additional inputs and is
trained on `lambda in {{0, 0.25, 0.5, 0.75, 1}}` with `alpha = 0.8`. Evaluation
also includes held-out interpolation points `lambda in {{0.125, 0.375, 0.625,
0.875}}`.

Key numerical summary:

- Conditioned model, training lambdas: average mean cost `{train_mean:.4f}`,
  average `ES_0.8` `{train_es:.4f}`, average variance `{train_var:.4f}`.
- Separate fixed models, same lambdas: average mean cost `{fixed_train_mean:.4f}`,
  average `ES_0.8` `{fixed_train_es:.4f}`, average variance `{fixed_train_var:.4f}`.
- Conditioned-minus-fixed gaps on the training lambdas: mean cost
  `{mean_gap:+.4f}`, `ES_0.8` `{es_gap:+.4f}`, variance `{var_gap:+.4f}`.
- Held-out conditioned lambdas: average mean cost `{heldout_mean:.4f}`,
  average `ES_0.8` `{heldout_es:.4f}`, average variance `{heldout_var:.4f}`.
- Across all conditioned lambda values, `ES_0.8` ranges from
  `{best_es_row['ES_0.8']:.4f}` at `lambda={best_es_row['lambda']:.3g}` to
  `{worst_es_row['ES_0.8']:.4f}` at `lambda={worst_es_row['lambda']:.3g}`.

Short proposal paragraph:

As a second-stage prototype, I implemented a preference-conditioned composite
risk-sensitive actor-critic. Instead of training a separate actor and critic for
each fixed risk preference, one policy is trained as
`pi_phi(a | s, lambda, alpha)` and one critic as `V_psi(s, lambda, alpha)`.
For the first controlled version, I use a discrete training grid over lambda and
maintain a separate VaR auxiliary estimate `v_star(lambda, alpha)` for each
training preference. This keeps the implementation faithful to the convex
scoring-function construction while testing whether a single shared model can
represent a family of composite risk preferences.

In the five-seed experiment, the conditioned actor-critic reaches performance
close to the separately trained fixed-composite policies on the training lambda
grid and gives stable interpolation behavior on held-out lambda values. The
held-out points have average `ES_0.8` `{heldout_es:.4f}`, close to the training
grid average `{train_es:.4f}`, suggesting that the learned policy family varies
smoothly with the risk-preference input. These results support the feasibility
of moving from a unified scoring representation for individual risk objectives
to a unified preference-conditioned policy representation.

Cautious wording:

The current evidence should be described as a discrete preference-conditioned
prototype, not yet a fully continuous alpha/lambda model. The first experiment
fixes `alpha = 0.8` and tests interpolation only along the lambda dimension.
Future work can replace the table-based `v_star(lambda, alpha)` with a learned
or binned auxiliary model and extend training to continuous preference sampling.
"""
    (OUTPUT_DIR / "proposal_conditioned_text.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    fixed_rows = read_fixed_rows()
    conditioned_rows = read_conditioned_rows()
    save_conditioned_table(conditioned_rows)
    save_train_grid_comparison_table()
    save_metric_comparison(fixed_rows, conditioned_rows)
    save_frontier(fixed_rows, conditioned_rows)
    save_train_grid_gap(fixed_rows, conditioned_rows)
    save_proposal_text(fixed_rows, conditioned_rows)
    print(f"Wrote conditioned figures to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
