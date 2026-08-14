"""Publication-quality experiment figures (SPEC E-5).

All figures use the Agg backend (headless-safe), consistent typography, and
are written as 200-dpi PNGs plus vector SVGs.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from origin_ml.evaluation.experiment import ExperimentResult
    from origin_ml.evaluation.metrics import CalibrationBin

__all__ = ["write_experiment_plots"]

_STYLE = {
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
}

_HUMAN_COLOR = "#3b7dd8"
_AI_COLOR = "#d85f3b"
_NEUTRAL = "#6b7280"


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png")
    fig.savefig(out_dir / f"{name}.svg")
    plt.close(fig)


def _plot_ablation_bars(result: ExperimentResult, out_dir: Path) -> None:
    names = list(result.ablations)
    seen_auroc = [result.ablations[n].seen.auroc or 0.0 for n in names]
    loc_auroc = [result.ablations[n].localization.auroc or 0.0 for n in names]

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    x = range(len(names))
    width = 0.38
    ax.bar(
        [i - width / 2 for i in x], seen_auroc, width, label="document (seen)", color=_HUMAN_COLOR
    )
    ax.bar(
        [i + width / 2 for i in x], loc_auroc, width, label="sentence localization", color=_AI_COLOR
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels([n.replace("_", "\n") for n in names])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC")
    ax.set_title(f"Ablation comparison — {result.name}")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, out_dir, "ablation_auroc")


def _plot_robustness(result: ExperimentResult, out_dir: Path) -> None:
    names = list(result.ablations)
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    x = range(len(names))
    width = 0.28
    seen = [result.ablations[n].seen.f1 for n in names]
    para = []
    for n in names:
        paraphrased = result.ablations[n].paraphrased
        para.append(paraphrased.f1 if paraphrased is not None else 0.0)
    unseen_means = []
    for n in names:
        unseen = result.ablations[n].unseen_family
        unseen_means.append(sum(m.f1 for m in unseen.values()) / len(unseen) if unseen else 0.0)
    ax.bar([i - width for i in x], seen, width, label="seen", color=_HUMAN_COLOR)
    ax.bar(list(x), para, width, label="paraphrased", color=_NEUTRAL)
    ax.bar([i + width for i in x], unseen_means, width, label="unseen family", color=_AI_COLOR)
    ax.set_xticks(list(x))
    ax.set_xticklabels([n.replace("_", "\n") for n in names])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title("Robustness: seen vs paraphrased vs unseen model family")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, out_dir, "robustness_f1")


def _plot_calibration(
    bins: tuple[CalibrationBin, ...], title: str, out_dir: Path, name: str
) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color=_NEUTRAL, linewidth=1, label="perfect")
    if bins:
        ax.plot(
            [b.mean_predicted for b in bins],
            [b.frac_positive for b in bins],
            marker="o",
            color=_AI_COLOR,
            label="observed",
        )
    ax.set_xlabel("mean predicted P(ai)")
    ax.set_ylabel("observed AI fraction")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left")
    _save(fig, out_dir, name)


def write_experiment_plots(result: ExperimentResult, out_dir: Path) -> list[Path]:
    """Write all figures for an experiment; returns the created file paths."""
    with plt.rc_context(_STYLE):
        _plot_ablation_bars(result, out_dir)
        _plot_robustness(result, out_dir)
        if "classical_full" in result.ablations:
            _plot_calibration(
                result.ablations["classical_full"].seen.calibration_bins,
                "Calibration — classical_full (seen docs)",
                out_dir,
                "calibration_classical_full",
            )
    return sorted(out_dir.glob("*.png")) + sorted(out_dir.glob("*.svg"))
