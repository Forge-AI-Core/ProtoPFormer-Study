# ProtoPFormer-Study

[![Paper](https://img.shields.io/badge/Paper-ECCV%202022-blue)](https://arxiv.org/abs/2208.10431)
[![Original](https://img.shields.io/badge/Original-zju--vipa%2FProtoPFormer-black)](https://github.com/zju-vipa/ProtoPFormer)

ProtoPFormer adds **Global + Local prototype branches** on top of DeiT, giving
**intrinsic interpretability** ("this looks like that") for fine-grained recognition.
This study repo extends the team's fine-grained pool from **post-hoc attention**
(TransFG / PIM / RA-CNN) to **intrinsic prototype reasoning**, then transfers to the
**Bogonet industrial dataset (Phase 2, private)**.

## 👁️ Project Goal
Make each prediction **auditable** — the model points to the image region (prototype)
behind its decision. Target: safety-critical industrial inspection where *why* matters.

## 🚀 Objective
```
ImageNet (DeiT-S/16)  →  Bogonet 3-class domain transfer (our own train/val split)  →  interpretable classifier
```

This branch (**SCRUM-82**) delivers **test set evaluation** on a newly received test set (`Bogonet_data/testset/`) for crops 25pct + 0pct, using best checkpoints from SCRUM-48/56 with **val-derived thresholds**. PR curves, per-class metrics, and confusion matrices for each crop are added under `result/<crop>/test/`.
Sibling tickets: SCRUM-48 (25pct train, merged), SCRUM-52 (10pct train, merged), SCRUM-56 (0pct train, merged), SCRUM-81 (TransFG parallel test eval).

---

## 1. Setup (Python 3.12 · uv)

Reproducible environment via [uv](https://docs.astral.sh/uv/):

```bash
# 1) install uv (once per machine)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) reproduce the exact environment (creates .venv from uv.lock)
uv sync

# 3) run anything inside that environment
uv run python main.py ...
```

- **Python**: `3.12` — pinned via `.python-version`
- **PyTorch**: `torch==2.10.0+cu130`, `torchvision==0.25.0+cu130` from the PyTorch **CUDA 13.0** index
  (declared in `pyproject.toml [tool.uv.sources]`, so `uv sync` fetches the correct aarch64 wheels)
- **Hardware (dev)**: NVIDIA GB10 (DGX Spark, aarch64, CUDA 13.0)
- Exact versions are frozen in **`uv.lock`** (committed) → identical env for every teammate.

> ℹ️ Scripts/identifiers use the `bogonet` name throughout.

---

## 2. 🏋️ Training (this branch: crop=0pct)

```bash
# bash scripts/train_bogonet.sh <batch> <epochs> <expansion>
uv run bash scripts/train_bogonet.sh 64 200 0pct
```
- **Class imbalance** → `--balanced_sampler` (WeightedRandomSampler, train loader only)
- Cosine schedule + warmup; **best checkpoint selected by balanced accuracy**
- Resume a stopped run: pass a 4th arg (a `checkpoint-latest.pth` path)
- Full hyperparameters → [`result/0pct/parameters.md`](result/0pct/parameters.md)

## 3. 📊 Result — 0pct crop, 200 epochs

Headline metric = **balanced accuracy** (mean per-class recall — the meaningful metric under imbalance), plus per-class **precision / recall / F1** and a **confusion matrix**.

| metric | value |
| :--- | :--- |
| balanced accuracy (best) | **71.59%** |
| per-class recall (cut / danger / excluded) | 0.80 / 0.71 / 0.64 |
| per-class precision (cut / danger / excluded) | 0.80 / 0.67 / 0.70 |
| per-class F1 (cut / danger / excluded) | 0.80 / 0.69 / 0.67 |

**Crop-expansion ablation outlook** (same split02, 200 epochs):

| crop | balanced acc | ticket |
| :--- | ---: | :--- |
| 25pct | 73.16% | SCRUM-48 |
| 10pct | 70.94% | SCRUM-52 |
| **0pct** (this branch) | **71.59%** | SCRUM-56 |

→ ProtoPFormer crop 패턴 = **25 > 0 > 10 (비단조)**. tight crop(0pct, 컨텍스트 없음)에도 71.59%로 **강건** — local prototype 매칭이 특정 patch 중심이라 컨텍스트에 덜 의존. TransFG(전역 attention)는 0pct에서 67%로 급락해 대비됨.

**Train/Val overfit check (0pct):**

![Train/Val overfit check — 0pct](result/0pct/overfit_curve.png)

Full artifacts in this branch: [`result/0pct/`](result/0pct/) — `metrics.md` (P/R/F1+confusion), `parameters.md` (hyperparams), `train_val_history.md` + `overfit_curve.png` (overfit check).

## 4. 🔥 Prototype Visualization ★ (core deliverable)

```bash
uv run bash scripts/visualize_bogonet.sh epoch-best.pth
```
Generates per-prototype **activation heatmaps**: *which patch matched which class prototype*,
making each decision auditable.

> "Why was this sample classified as class X?" → "Patch Y matches prototype P_k of class X."

*(Sample images are omitted here — private dataset.)*

## 5. 📈 Output & Metrics layout

```
output/.../checkpoints/epoch-best.pth        # best by balanced_acc
output/.../checkpoints/checkpoint-latest.pth # per-epoch, for --resume
output/.../tf-logs/                          # TensorBoard scalars
output/.../train-logs/                       # text log (per-epoch metrics, confusion matrix)
```

## 6. 🧪 Test Set Evaluation (SCRUM-82)

Newly received test set (`Bogonet_data/testset/`, **326 samples per crop**) evaluated with best checkpoints from SCRUM-48 (25pct) and SCRUM-56 (0pct). **Validation-derived threshold** (rule: `P(danger) ≥ τ → danger; else argmax({cut, excluded})`, τ selected to maximize val balanced_acc) applied to test predictions; argmax baseline reported alongside.

> **Test set distribution differs sharply from train/val** — `cut 33 / danger 266 / excluded 27` (danger **82 %**), simulating realistic operational deployment where most inspected items warrant review. Absolute accuracy looks low because danger errors dominate the count; PR-AUC for danger is the meaningful signal.

### Test crop ablation (argmax)

| crop | val τ\* | test acc | test bacc | danger P | danger R | danger AP | mAP |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25pct | 0.45 | 38.96 % | 34.30 % | 78.26 % | 40.60 % | **0.785** | 0.329 |
| 0pct  | 0.45 | **39.26 %** | **36.19 %** | 78.10 % | 40.23 % | **0.786** | **0.342** |

Per-class details + confusion matrices + PR curves: [`result/25pct/test/`](result/25pct/test/) · [`result/0pct/test/`](result/0pct/test/)

### Key observations

- **Danger PR-AUC ≈ 0.785** for both crops — model **discriminates danger well** in confidence ranking, but operating-point recall is conservative.
- **Danger precision ~78 %** at argmax — when ProtoPFormer says danger, it's usually right.
- **Danger recall only ~40 %** at argmax — **misses ~60 %** of dangers, far below TransFG (~58 %). prototype voting under operational distribution is overly cautious — many dangers classified as `cut`.
- **Val → test threshold portability OK**: τ\* derived on val (= 0.45 for both crops) gives **small positive lift** on test (+0.5–1.2 %p acc), unlike TransFG where threshold transfer was slightly negative.
- **Crop effect inverted vs val**: val showed 25pct > 0pct, but **test shows 0pct slightly better** for ProtoPFormer (acc 39.26 vs 38.96, mAP 0.342 vs 0.329). Local prototype matching is less crop-sensitive than expected.
- **Cut + excluded AP very low** (0.09–0.13): minorities in test (33 + 27 of 326), one-vs-rest PR is penalized; per-class signal limited.

### How to reproduce

```bash
proto_venv/bin/python test_bogonet.py --expansion 25pct
proto_venv/bin/python test_bogonet.py --expansion 0pct
```

Loads best `epoch-best.pth` checkpoint, runs val for τ sweep (saves `threshold_sweep.md`), then test inference. Outputs `metrics.md`, `pr_curve.png`, `confusion_matrix.png`, `threshold_sweep.md` to `result/<crop>/test/`.

> Sibling SCRUM-81 (TransFG-Bogo) ran identical experiment for cross-model comparison: TransFG showed higher test danger recall (~58 %) but similar danger AP (~0.82). See sibling repo PR for details.

---

## 🔧 Modifications from Original

| Area | Change | Why |
| :--- | :--- | :--- |
| timm 1.0 compat | pop meta kwargs in `MyVisionTransformer`; optional CaiT import | run on timm 1.0 |
| Class imbalance | `--balanced_sampler` (WeightedRandomSampler) | minority classes |
| Metric | balanced accuracy + per-class confusion matrix | imbalanced eval |
| Checkpointing | best = balanced_acc; per-epoch `checkpoint-latest` (resume) | recall-first / robust long runs |
| Crop ablation | `--expansion` arg (0 / 10 / 25 %) | context-vs-tightness study |
| Test eval (SCRUM-82) | `test_bogonet.py` — val τ sweep → val-τ\* on test + argmax + PR curves | apply val operating point on real-distribution test set |
| torch 2.6 / mpl 3.4 | `torch.load(weights_only=False)`; `fig.add_subplot(projection='3d')` | visualization on current stack |
| Hardware | NVIDIA GB10 / aarch64 / CUDA 13 single-GPU path | dev machine |

## ♻️ Reproducibility Policy

Any teammate should read this README, run the committed files as-is (`uv sync` → `uv run ...`),
and obtain the same result. Fixed seed (1028); environment frozen in `uv.lock`; results recorded above;
no aspirational commands.

## Citation

```bibtex
@inproceedings{xue2022protopformer,
  title={ProtoPFormer: Concentrating on Prototypical Parts in Vision Transformers for Interpretable Image Recognition},
  author={Xue, Mengqi and Huang, Qihan and Zhang, Haofei and Cheng, Lechao and Song, Jie and Wu, Minghui and Song, Mingli},
  booktitle={ECCV},
  year={2022}
}
```

### Acknowledgement
- Official ProtoPFormer implementation: [zju-vipa/ProtoPFormer](https://github.com/zju-vipa/ProtoPFormer)
- Team fine-grained research track: [Forge-AI-Core](https://github.com/Forge-AI-Core)
