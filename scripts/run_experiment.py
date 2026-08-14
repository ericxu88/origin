"""Run an Origin experiment from a JSON config (SPEC E-6).

Usage::

    uv run python scripts/run_experiment.py experiments/sample.json
"""

from __future__ import annotations

import sys
from pathlib import Path

from origin_ml.evaluation import ExperimentConfig, run_experiment


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: run_experiment.py <config.json>", file=sys.stderr)
        return 2
    config = ExperimentConfig.from_file(Path(argv[0]))
    result = run_experiment(config)
    out_dir = Path(config.output_dir) / config.name
    print(f"experiment {result.name!r}: {result.n_train} train / {result.n_test} test docs")
    for name, ablation in result.ablations.items():
        auroc = f"{ablation.seen.auroc:.3f}" if ablation.seen.auroc is not None else "n/a"
        loc = (
            f"{ablation.localization.auroc:.3f}"
            if ablation.localization.auroc is not None
            else "n/a"
        )
        print(f"  {name:22s} seen AUROC={auroc}  localization AUROC={loc}")
    if result.rq6_correlations:
        corr = result.rq6_correlations.get("neural_vs_classical_prob")
        if corr is not None:
            print(f"  RQ6: corr(neural, classical) = {corr:.3f}")
    print(f"outputs: {out_dir}/results.json + figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
