# ProtoPFormer 10pct — metrics (Bogonet 3-class)

- crop **10pct**, DeiT-S/16, 200 epoch, split02 val 3,369, WeightedRandomSampler
- checkpoint: `epoch-best.pth` (best balanced_acc @ epoch 190)
- **balanced accuracy = 70.94%**, acc1 = 74.86%

## per-class (precision / recall / f1, %)
| class | precision | recall | f1 |
|---|---:|---:|---:|
| cut | 79.6 | 81.9 | 80.7 |
| danger | 68.1 | 69.5 | 68.8 |
| excluded | 70.6 | 61.4 | 65.7 |
| **avg** | **72.8** | **70.94** | **71.7** |

## confusion matrix (행=정답, 열=예측)
| true\pred | cut | danger | excluded |
|---|---:|---:|---:|
| cut | 1493 | 242 | 87 |
| danger | 241 | 681 | 58 |
| excluded | 142 | 77 | 348 |

## note
- danger recall {R[1]:.1f}% (FN: danger→cut {cm[1,0]}, danger→excluded {cm[1,2]}).
- 과적합 추이: train_val_history.md / overfit_curve.png
