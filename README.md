# ProtoPFormer-Study

[![Paper](https://img.shields.io/badge/Paper-ECCV%202022-blue)](https://arxiv.org/abs/2208.10431)
[![Original](https://img.shields.io/badge/Original-zju--vipa%2FProtoPFormer-black)](https://github.com/zju-vipa/ProtoPFormer)

ProtoPFormer adds **Global + Local prototype branches** on top of DeiT, giving
**intrinsic interpretability** ("this looks like that") for fine-grained recognition.
This study repo extends the team's fine-grained pool from **post-hoc attention**
(TransFG / PIM / RA-CNN) to **intrinsic prototype reasoning**, then transfers to the
**Vogonet industrial dataset (Phase 2, private)**.

## 👁️ Project Goal
Make each prediction **auditable** — the model points to the image region (prototype)
behind its decision. Target: safety-critical industrial inspection where *why* matters.

## 🚀 Objective
```
ImageNet (DeiT-S/16)  →  Vogonet 3-class domain transfer (our own train/val split)  →  interpretable classifier
```

---

## 1. Setup (Python 3.12 · uv)

Reproducible environment via [uv](https://docs.astral.sh/uv/). A teammate only needs:

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
  (declared in `pyproject.toml [tool.uv.sources]`, so `uv sync` fetches the correct aarch64 wheels automatically)
- **Hardware (dev)**: NVIDIA GB10 (DGX Spark, aarch64, CUDA 13.0)
- Exact versions are frozen in **`uv.lock`** (committed) → identical env for every teammate.

> ℹ️ Scripts/identifiers use the `vogonet` name (aligned with this repo's public alias).

---

## 2. 🏋️ Training

```bash
# bash scripts/train_vogonet.sh <batch> <epochs> <crop_expansion>
uv run bash scripts/train_vogonet.sh 64 200 25pct
```
- **Class imbalance** → `--balanced_sampler` (WeightedRandomSampler, train loader only)
- cosine schedule + warmup; **best checkpoint is selected by balanced accuracy**
- resume a stopped run: pass a 4th arg (a `checkpoint-latest.pth` path)

## 3. 📊 Evaluation & Inference

Reported with **balanced accuracy** (mean per-class recall — the meaningful metric under imbalance),
plus per-class **precision / recall / F1** and a **confusion matrix**.

**Result (Vogonet 3-class, our own split, DeiT-S/16, 200 epochs):**

| metric | value |
| :--- | :--- |
| balanced accuracy (best) | **73.16%** |
| per-class recall | 0.79 / 0.71 / 0.70 |

Crop-expansion ablation (same split, 200ep): 25% **73.16%** > 10% 70.94% — surrounding context
helps the harder minority class, while the self-evident class prefers tighter crops.

## 4. 🔥 Prototype Visualization ★ (core deliverable)

```bash
uv run bash scripts/visualize_vogonet.sh epoch-best.pth
```
Generates per-prototype **activation heatmaps**: *which patch matched which class prototype*,
making each decision auditable.

> "Why was this sample classified as class X?" → "Patch Y matches prototype P_k of class X."

*(Sample images are omitted here — private dataset.)*

## 5. 📈 Output & Metrics

```
output/.../checkpoints/epoch-best.pth        # best by balanced_acc
output/.../checkpoints/checkpoint-latest.pth # per-epoch, for --resume
output/.../tf-logs/                          # TensorBoard scalars
output/.../train-logs/                       # text log (per-epoch metrics, confusion matrix)
```

---

## 🔧 Modifications from Original

| Area | Change | Why |
| :--- | :--- | :--- |
| timm 1.0 compat | pop meta kwargs in `MyVisionTransformer`; optional CaiT import | run on timm 1.0 |
| Class imbalance | `--balanced_sampler` (WeightedRandomSampler) | minority classes |
| Metric | balanced accuracy + per-class confusion matrix | imbalanced eval |
| Checkpointing | best = balanced_acc; per-epoch `checkpoint-latest` (resume) | recall-first / robust long runs |
| Crop ablation | `--expansion` arg (0 / 10 / 25 %) | context-vs-tightness study |
| torch 2.6 / mpl 3.4 | `torch.load(weights_only=False)`; `fig.add_subplot(projection='3d')` | visualization on current stack |
| Hardware | NVIDIA GB10 / aarch64 / CUDA 13 single-GPU path | dev machine |

## 🔗 Related Repos (Forge-AI-Core)

| Repo | Paradigm | Backbone |
| :--- | :--- | :--- |
| [TransFG-Study](https://github.com/Forge-AI-Core/TransFG-Study) | Post-hoc PSM | ViT-B/16 |
| [FGVC-PIM](https://github.com/Forge-AI-Core/FGVC-PIM) | Plug-in Module | Swin-T |
| [FGVC-RA-CNN](https://github.com/Forge-AI-Core/FGVC-RA-CNN) | Recurrent Attention | VGG/ResNet |
| **ProtoPFormer-Study** | **Intrinsic Prototype** | **DeiT-S/16** |

## ♻️ Reproducibility Policy

Any teammate should read this README, run the committed files as-is (`uv sync` → `uv run ...`),
and obtain the same result. Fixed seed; environment frozen in `uv.lock`; results recorded as numbers above;
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
