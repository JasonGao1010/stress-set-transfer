# Method and data

## Transfer quantities

Let `S` be the `k` scenes selected using a source detector, `T` the target detector's top-`k` scenes, and `l(s)` the positive target loss in scene `s`.

| Quantity | Definition | Interpretation |
|---|---|---|
| Identity agreement | `|S ∩ T| / k` | Fraction of target hard scenes also selected by the source |
| Target-loss coverage | `sum(l(s), s in S) / sum(l(s), all s)` | Fraction of total target loss captured |
| Target-oracle efficiency | `sum(l(s), s in S) / sum(l(s), s in T)` | Loss capture relative to a target-specific selection at the same budget |

The outcome definitions are NDS-like scene loss, the fraction of clean true positives lost after perturbation, and the count of those lost true positives. Top-k boundary ties are averaged exactly using inclusion probabilities. Conditional references preserve the observed acquisition-log composition; the lost-count opportunity reference additionally conditions on clean-detection opportunities within each log.

For a target with two available non-target source sets, let `E_best` denote the higher target-oracle efficiency and `E_selected` the efficiency of the source chosen by scene overlap. The exact decomposition is:

```text
Total reuse regret = 1 - E_selected
Availability regret = 1 - E_best
Selection regret = E_best - E_selected
```

The 87.6–96.3% result uses the ratio of mean availability regret to mean total reuse regret, raw scene overlap for source choice, and the 20% budget. Its six values correspond to three outcomes in each of two partitions. The selection diagnostic uses observed target outcomes.

## Detector and dataset provenance

The study evaluates public models from [BEVFusion / MMDetection3D](https://github.com/open-mmlab/mmdetection3d), [SparseFusion](https://github.com/yichen928/SparseFusion), and [DeepInteraction](https://github.com/fudan-zvg/DeepInteraction) on [nuScenes](https://www.nuscenes.org/). Scene membership and acquisition-log identifiers are recorded in [data/scene_split.json](data/scene_split.json).

The development partition contains 50 scenes, 2,007 keyframes, and nine logs. The follow-up contains 18 scenes, 725 keyframes, and nine separate logs. Development evaluates clean calibration and ±1° roll, pitch, yaw plus ±0.10 m translations. The follow-up evaluates clean calibration and the six signed rotations. The rotation family was selected after development analysis.

Targets are matched to detections through class-aware global one-to-one assignment: legal edges require equal class, score ≥0.25, and center distance ≤2 m; the assignment first maximizes cardinality, then minimizes total distance. Eligibility is defined by a clean true-positive match. A failure occurs when that target loses its qualifying match under perturbation. These are the study's target-transition definitions.

## Released inputs

The `data/confirmation/` directory mirrors the development inputs for the follow-up partition.

| File | Contents |
|---|---|
| `primary_checkpoint.json` | Per-scene detector losses and acquisition-log membership |
| `mechanism.json` | Detector-derived failure summaries and matching configuration |
| `target_table.parquet` | One row per detector / condition / eligible target, with scene and log identifiers |
| `target_sensitivity.parquet` | Eligibility and failure bit masks for nine score/radius configurations |
| `scene_split.json` | Metadata-based partition selection and exact scene membership |

The target table includes detector, condition, log, scene, sample token, annotation token, class, clean score, distance, volume, visibility, and binary failure. The compact sensitivity table covers scores 0.20, 0.25, and 0.30 crossed with radii 1.5, 2.0, and 2.5 m. nuScenes tokens retain the correspondence between records across detectors and conditions.

The code's MIT license covers the original analysis implementation. Dataset-derived records and displayed nuScenes imagery retain their upstream terms. The repository distributes the analysis inputs and results; raw camera images, point clouds, and detector weights are obtained from their respective sources.
