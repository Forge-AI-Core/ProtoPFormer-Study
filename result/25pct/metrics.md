# ProtoPFormer 25pct — metrics (Bogonet 3-class)

- model: ProtoPFormer (DeiT-S/16, 224), crop expansion **25pct**, 200 epoch
- data: split02 val 3,369 (cut 1,822 / danger 980 / excluded 567), WeightedRandomSampler
- checkpoint: `epoch-best.pth` (best balanced_acc @ epoch 128)
- **balanced accuracy (avg recall) = 73.16%**, acc1 = 75.10%

## per-class (precision / recall / f1, %)
| class | precision | recall | f1 |
|---|---:|---:|---:|
| cut | 81.6 | 79.2 | 80.4 |
| danger | 68.7 | 70.6 | 69.6 |
| excluded | 66.7 | 69.7 | 68.2 |
| **avg** | **72.3** | **73.16** | **72.7** |

## confusion matrix (행=정답, 열=예측, val 절대 개수)
| true\pred | cut | danger | excluded |
|---|---:|---:|---:|
| cut | 1443 | 255 | 124 |
| danger | 215 | 692 | 73 |
| excluded | 111 | 61 | 395 |

## 정의
- recall = TP/(TP+FN) 실제 중 맞힌 비율 (놓침↓). danger에서 안전상 핵심.
- precision = TP/(TP+FP) 예측 중 맞은 비율 (오경보↓).
- f1 = 2·P·R/(P+R), balanced accuracy = recall 평균.

## note
- danger recall 70.6% → 위험물 약 29% 놓침(FN: danger→cut 215, danger→excluded 73) → 안전상 추가 개선 과제.
- 과적합 추이: `train_val_history.md` / `overfit_curve.png` 참조.
