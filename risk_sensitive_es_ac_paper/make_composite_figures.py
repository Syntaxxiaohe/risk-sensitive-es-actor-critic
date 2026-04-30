"""Build proposal-ready figures and text for the composite ES prototype."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


GRID_ROOT = Path("multirun_composite_grid_1200x8192")
BASELINE_SUMMARY = Path("multirun_batched_1200x8192") / "summary_by_model.csv"
OUTPUT_DIR = Path("composite_figures")


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_composite_rows() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    lambda_tags = [
        (0.0, "l0"),
        (0.25, "l0p25"),
        (0.5, "l0p5"),
        (0.75, "l0p75"),
        (1.0, "l1"),
    ]
    for risk_lambda, tag in lambda_tags:
        path = GRID_ROOT / f"summary_by_model_composite_es_{tag}.csv"
        model = f"composite_es_{tag}_best"
        match = [row for row in read_summary(path) if row["model"] == model]
        if not match:
            raise RuntimeError(f"Missing {model} in {path}")
        row = match[0]
        rows.append(
            {
                "lambda": risk_lambda,
                "mean_cost": _float(row, "mean_cost_mean"),
                "mean_cost_std": _float(row, "mean_cost_std"),
                "ES_0.8": _float(row, "ES_0.8_mean"),
                "ES_0.8_std": _float(row, "ES_0.8_std"),
                "ES_0.6": _float(row, "ES_0.6_mean"),
                "ES_0.6_std": _float(row, "ES_0.6_std"),
                "variance": _float(row, "variance_mean"),
                "variance_std": _float(row, "variance_std"),
                "mean_var_utility": _float(row, "mean_var_utility_mean"),
                "mean_var_utility_std": _float(row, "mean_var_utility_std"),
            }
        )
    return rows


def read_baselines() -> dict[str, dict[str, float]]:
    baselines: dict[str, dict[str, float]] = {}
    if not BASELINE_SUMMARY.exists():
        return baselines
    wanted = {"mean_best", "es_best", "es06_best", "mean_var_best", "always_zero"}
    for row in read_summary(BASELINE_SUMMARY):
        model = row["model"]
        if model not in wanted:
            continue
        baselines[model] = {
            "mean_cost": _float(row, "mean_cost_mean"),
            "ES_0.8": _float(row, "ES_0.8_mean"),
            "ES_0.6": _float(row, "ES_0.6_mean"),
            "variance": _float(row, "variance_mean"),
        }
    return baselines


def save_table(rows: list[dict[str, float]], baselines: dict[str, dict[str, float]]) -> None:
    csv_path = OUTPUT_DIR / "composite_lambda_table.csv"
    fields = [
        "lambda",
        "mean_cost",
        "mean_cost_std",
        "ES_0.8",
        "ES_0.8_std",
        "ES_0.6",
        "ES_0.6_std",
        "variance",
        "variance_std",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)

    md_lines = [
        "# Composite ES Lambda Grid",
        "",
        "| lambda | mean cost | ES0.8 | ES0.6 | variance |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['lambda']:.2f} | {row['mean_cost']:.4f} | {row['ES_0.8']:.4f} | "
            f"{row['ES_0.6']:.4f} | {row['variance']:.4f} |"
        )
    if baselines:
        md_lines.extend(["", "Reference single-objective baselines:", ""])
        md_lines.append("| model | mean cost | ES0.8 | ES0.6 | variance |")
        md_lines.append("|---|---:|---:|---:|---:|")
        for name in ["mean_best", "es_best", "es06_best", "mean_var_best"]:
            if name not in baselines:
                continue
            row = baselines[name]
            md_lines.append(
                f"| {name} | {row['mean_cost']:.4f} | {row['ES_0.8']:.4f} | "
                f"{row['ES_0.6']:.4f} | {row['variance']:.4f} |"
            )
    (OUTPUT_DIR / "composite_lambda_table.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


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


def save_metric_panel(rows: list[dict[str, float]]) -> None:
    lambdas = [row["lambda"] for row in rows]
    metrics = [
        ("mean_cost", "Mean Cost", "#2F6BFF"),
        ("ES_0.8", "ES 0.8", "#D94E41"),
        ("ES_0.6", "ES 0.6", "#D29021"),
        ("variance", "Variance", "#2A9D77"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.6), constrained_layout=True)
    fig.suptitle("Composite Mean-ES Objective: Lambda Controls the Risk Tradeoff", y=1.02)
    for ax, (key, label, color) in zip(axes.ravel(), metrics):
        values = [row[key] for row in rows]
        stds = [row[f"{key}_std"] for row in rows]
        ax.errorbar(lambdas, values, yerr=stds, color=color, marker="o", linewidth=2.0, capsize=3)
        ax.set_xlabel("lambda")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        ax.set_xticks(lambdas)
    fig.savefig(OUTPUT_DIR / "composite_lambda_metrics.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "composite_lambda_metrics.pdf", bbox_inches="tight")
    plt.close(fig)


def save_frontier(rows: list[dict[str, float]], baselines: dict[str, dict[str, float]]) -> None:
    mean = [row["mean_cost"] for row in rows]
    es08 = [row["ES_0.8"] for row in rows]
    lambdas = [row["lambda"] for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    ax.plot(mean, es08, color="#1F5E7A", marker="o", linewidth=2.2, label="composite_es grid")
    for x, y, lmbda in zip(mean, es08, lambdas):
        ax.annotate(f"{lmbda:g}", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)

    baseline_styles = {
        "mean_best": ("Mean baseline", "#7A7A7A", "s"),
        "es_best": ("ES baseline", "#D94E41", "D"),
        "es06_best": ("ES0.6 baseline", "#D29021", "^"),
        "mean_var_best": ("Mean-Var baseline", "#2A9D77", "v"),
    }
    for key, (label, color, marker) in baseline_styles.items():
        if key not in baselines:
            continue
        row = baselines[key]
        ax.scatter(row["mean_cost"], row["ES_0.8"], color=color, marker=marker, s=65, label=label, zorder=4)

    ax.set_title("Risk-Return Frontier Induced by Composite Preferences")
    ax.set_xlabel("Mean cost (lower is better)")
    ax.set_ylabel("ES 0.8 total-cost tail risk (lower is better)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.savefig(OUTPUT_DIR / "composite_frontier_mean_vs_es08.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "composite_frontier_mean_vs_es08.pdf", bbox_inches="tight")
    plt.close(fig)


def save_proposal_text(rows: list[dict[str, float]]) -> None:
    first = rows[0]
    last = rows[-1]
    es_drop = 100.0 * (first["ES_0.8"] - last["ES_0.8"]) / first["ES_0.8"]
    var_drop = 100.0 * (first["variance"] - last["variance"]) / first["variance"]
    mean_increase = last["mean_cost"] - first["mean_cost"]
    text = f"""# Proposal Figure/Table Draft: Composite ES Prototype

Suggested caption:

Figure X reports a fixed composite risk objective
`(1 - lambda) E[C] + lambda ES_0.8(C)` trained with the same actor-critic
implementation, random seeds, and evaluation protocol as the paper-aligned
reproduction. Increasing `lambda` shifts the learned policy from mean-cost
optimization toward total-cost tail-risk control.

Key result text:

As `lambda` increases from 0 to 1, the average out-of-sample cost rises from
`{first['mean_cost']:.4f}` to `{last['mean_cost']:.4f}`, while `ES_0.8` falls
from `{first['ES_0.8']:.4f}` to `{last['ES_0.8']:.4f}` and variance falls from
`{first['variance']:.4f}` to `{last['variance']:.4f}`. This corresponds to an
approximately `{es_drop:.2f}%` reduction in `ES_0.8` and `{var_drop:.2f}%`
reduction in variance, at a mean-cost increase of `{mean_increase:.4f}`. The
results support the preliminary claim that composite scoring objectives can
produce a controllable trade-off between expected performance and tail risk.

Short proposal paragraph:

I implemented a preliminary fixed-composite extension of the authors'
convex-scoring-function actor-critic framework. The objective combines expected
total cost and expected shortfall as
`J_{{lambda,alpha}}(pi) = (1-lambda) E[C^pi] + lambda ES_alpha(C^pi)`. In the
OU statistical arbitrage environment, a five-seed lambda grid shows a monotone
trade-off: larger risk weight increases mean cost slightly but reduces tail
risk and variance. This provides empirical motivation for the next step:
learning a preference-conditioned actor-critic `pi(a | s, lambda, alpha)` that
can represent a continuum of risk-sensitive policies instead of retraining one
policy for each fixed preference.
"""
    (OUTPUT_DIR / "proposal_composite_text.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    rows = read_composite_rows()
    baselines = read_baselines()
    save_table(rows, baselines)
    save_metric_panel(rows)
    save_frontier(rows, baselines)
    save_proposal_text(rows)
    print(f"Wrote composite figures to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
