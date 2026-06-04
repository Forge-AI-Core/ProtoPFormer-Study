# ProtoPFormer 0pct — metrics (Bogonet 3-class)

- crop **0pct**, DeiT-S/16, 200 epoch, split02 val 3,369, WeightedRandomSampler
- checkpoint: `epoch-best.pth` (best balanced_acc @ epoch 192)
- **balanced accuracy = 71.59%**, acc1 = 74.74%

## per-class (precision / recall / f1, %)
| class | precision | recall | f1 |
|---|---:|---:|---:|
| cut | 80.5 | 80.4 | 80.4 |
| danger | 67.2 | 70.5 | 68.8 |
| excluded | 69.6 | 63.8 | 66.6 |
| **avg** | **72.4** | **71.59** | **72.0** |

## confusion matrix (행=정답, 열=예측)
| true\pred | cut | danger | excluded |
|---|---:|---:|---:|
| cut | 1465 | 258 | 99 |
| danger | 230 | 691 | 59 |
| excluded | 126 | 79 | 362 |

## note
- danger recall {R[1]:.1f}% (FN: danger→cut {cm[1,0]}, danger→excluded {cm[1,2]}).
- 과적합 추이: train_val_history.md / overfit_curve.png
