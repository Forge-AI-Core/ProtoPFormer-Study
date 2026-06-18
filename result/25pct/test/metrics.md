# Test Eval — ProtoPFormer-Study 25pct crop

- Checkpoint: `/home/changilkim/Documents/aiffel_class/ProtoPFormer/output/bogonet/Bogonet/deit_small_patch16_224/1028-adamw-0.05-200-protopformer/checkpoints/epoch-best.pth`
- Val set    : split02 val (3369 장)
- Test set   : `testset/crops_25pct` (326 장)
- Test class counts (cut/danger/excluded): [33, 266, 27]  (※ danger 압도)

## Validation 에서 결정한 τ*

- **τ\* = 0.45** (val balanced_acc 최대)
- 규칙: `P(danger) ≥ τ → danger; 아니면 argmax({cut, excluded})`
- Val balanced_acc(τ\*) = 0.7345
- 참고 — F-beta(0.5) 최대 τ = 0.75, fbeta=0.7218

(threshold sweep 전체: [`threshold_sweep.md`](threshold_sweep.md))

## Test 결과 — A. argmax

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| cut      |   8.89 |  36.36 |  14.29 |    33 |
| danger   |  78.26 |  40.60 |  53.47 |   266 |
| excluded |  13.21 |  25.93 |  17.50 |    27 |

- Accuracy = **38.96%**
- Balanced acc = **34.30%**

### Confusion matrix (argmax)
```
              cut  danger  excluded
  cut         12     19        2
  danger     114    108       44
  excluded     9     11        7
```

## Test 결과 — B. τ\*=0.45 적용

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| cut      |   9.23 |  36.36 |  14.72 |    33 |
| danger   |  79.43 |  42.11 |  55.04 |   266 |
| excluded |  12.73 |  25.93 |  17.07 |    27 |

- Accuracy = **40.18%**
- Balanced acc = **34.80%**

### Confusion matrix (τ\* 적용)
```
              cut  danger  excluded
  cut         12     18        3
  danger     109    112       45
  excluded     9     11        7
```

## PR Curve (one-vs-rest, test)

| class | AP |
|---|---:|
| cut | 0.0883 |
| danger | 0.7848 |
| excluded | 0.1126 |
| **mean AP** | **0.3286** |

![PR curves](pr_curve.png)

## Confusion Matrix

![Confusion matrix](confusion_matrix.png)
