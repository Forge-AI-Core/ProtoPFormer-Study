# parameters — ProtoPFormer (crop 25pct)

> 학습 하이퍼파라미터(재현용). 실행: `scripts/train_bogonet.sh` → `main.py`. split02, leakage 0.

## 재현 명령
```bash
bash scripts/train_bogonet.sh 64 200 25pct
```

## 모델 / 초기화
| 항목 | 값 |
|---|---|
| backbone | deit_small_patch16_224 (DeiT-S/16) |
| 입력 해상도 | 224 |
| init | ImageNet pretrained (CUB warm-start 없음) |
| #params | ~22M |
| num_classes | 3 (cut/danger/excluded) |

## prototype
| 항목 | 값 |
|---|---|
| prototype_num | 30 (3 class × 10) |
| dim | 192 |
| use_global | True |
| global_proto_per_class | 10 |
| global_coe | 0.5 |
| reserve_layers | 11 |
| reserve_token_nums | 81 (9×9) |
| prototype_activation | log |

## 학습 (optimizer/scheduler)
| 항목 | 값 |
|---|---|
| batch_size | 64 |
| epochs | 200 |
| optimizer | AdamW |
| weight_decay | 0.05 |
| scheduler | cosine |
| warmup_epochs / warmup_lr | 5 / 1e-4 |
| features_lr | 1e-4 |
| add_on_layers_lr | 3e-3 |
| prototype_vectors_lr | 3e-3 |
| decay_epochs / rate | 10 / 0.1 |
| AMP(fp16) | 사용 |
| seed | 1028 |

## loss (실사용)
| 항목 | 계수 | 비고 |
|---|---|---|
| CrossEntropy | 1.0 | |
| PPC_σ (ppc_cov) | 0.1 | epoch ≥ 20 |
| PPC_μ (ppc_mean) | 0.5 | epoch ≥ 20 |
| ppc_cov_thresh / mean_thresh | 1.0 / 2.0 | |
> ※ cluster/separation/L1 계수는 args에만 존재, engine_proto에서 미사용.

## 불균형 / 평가
| 항목 | 값 |
|---|---|
| sampler | WeightedRandomSampler (train만) |
| best checkpoint 기준 | **balanced_acc** |

## augmentation (main.py default)
| color_jitter | 0.4 |
| auto_augment | rand-m9-mstd0.5-inc1 |
| random_erasing(reprob) | 0.25 |
| cutmix / mixup | 1.0 / 0.0 |
| label smoothing | 0.0 |
| drop_path | 0.1 |

## 데이터
| split | split02 (image-level, leakage 0) |
| crop | crops_25pct_split |
| val | 3,369 (cut 1,822 / danger 980 / excluded 567) |
