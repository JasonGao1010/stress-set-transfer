# Reproducing the analysis

The repository contains the detector-derived inputs needed to recompute scene-level transfer and item-level failure analysis. The computation uses CPU-based NumPy, pandas, PyArrow, SciPy, scikit-learn, and Matplotlib.

## Environment

Python 3.13 is the tested interpreter. Dependency versions are recorded in [requirements.txt](requirements.txt).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Scene-level transfer

```bash
python reproduce.py
```

For each partition, this command computes:

1. Top-10%, top-20%, and top-30% source scene sets with exact averaging over boundary ties.
2. Identity overlap, target-loss coverage, and target-oracle efficiency for three loss definitions.
3. Acquisition-log and clean-detection-opportunity references, using 10,000 conditional randomization replicates.
4. Source-choice regret decomposition and leave-one-log-out sensitivity.
5. An independent arithmetic reconstruction and comparison with the released primary summaries.

Outputs appear in `artifacts/`: `development.json`, `followup.json`, their Markdown tables, arithmetic-validation records, and `figures/transfer.png` / `transfer.svg` / `regret.csv`.

The independent validator reconstructs **216 development cells and 108 follow-up cells**, together with **216 and 108 source-choice checks**. The runner compares **1,022 numerical summary values** with the released results. The randomization seed is 20260724.

## Failure-sharing references and target-risk ranking

```bash
python reproduce.py --references
```

This adds analyses of the target-level Parquet tables: common clean-detection eligibility; fixed detector/condition/log failure margins; target-, family-, signed-axis-, and context-conditioned references; and cross-fitted source-to-target failure ranking. Reference-chain outputs contain convergence and mixing statistics. The held-log models fit all score transformations and learned parameters inside their training folds.

The additional outputs are `artifacts/development-references.json` and `artifacts/followup-references.json`, with corresponding Markdown summaries. This option has a longer runtime than scene-level analysis because it reconstructs item-level references and cross-fitted models.

## Figures

To redraw the repository figure directly from the released results:

```bash
python analysis/figures.py --results results --output assets
```

Panel A shows all 72 development detector-pair/perturbation cells for lost-clean-TP counts at a 20% scene budget; the diamonds mark medians. Panel B divides mean reuse regret into availability and selection components for each partition and loss definition. The CSV contains the exact values plotted in Panel B.

## Tests

```bash
python -m pytest -q
```

The 52 portable tests cover exact top-k ties, conditional references, coverage–efficiency identities, regret decomposition, failure-frequency constraints, class-aware matching metadata, and training-fold construction. The included derived-data corpus is the input to these analyses. Detector inference uses the original nuScenes data and the upstream detector implementations cited in [METHOD.md](METHOD.md).
