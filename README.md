# ProtoPFormer-Study

[![Paper](https://img.shields.io/badge/Paper-ECCV%202022-blue)](https://arxiv.org/abs/2208.10431)
[![Original Repo](https://img.shields.io/badge/Original-zju--vipa%2FProtoPFormer-black)](https://github.com/zju-vipa/ProtoPFormer)
[![Status](https://img.shields.io/badge/Status-WIP-yellow)](#-project-status)

**Paper**: [https://arxiv.org/abs/2208.10431](https://arxiv.org/abs/2208.10431)
**Original Repository**: [zju-vipa/ProtoPFormer](https://github.com/zju-vipa/ProtoPFormer)

ProtoPFormer proposes **Global + Local prototype branches** on top of DeiT, achieving competitive accuracy on fine-grained benchmarks with **intrinsic interpretability** ("this looks like that" reasoning).

This study repo extends the team's [fine-grained mechanism pool](https://github.com/Forge-AI-Core) from **post-hoc attention** (TransFG / PIM / RA-CNN) to **intrinsic prototype-based reasoning**, and will later transfer to Vogonet steel scrap data (Phase 2).

---

## 🚧 Project Status (as of 2026-05-18)

| Phase | Status | Note |
| :--- | :--- | :--- |
| Repo init (README, .gitignore) | ✅ Done | This commit |
| Environment (`pyproject.toml`, DeiT-S weights) | 📋 Planned | STEP 5 |
| CUB-200 training (smoke test) | 📋 Planned | STEP 6 |
| Aircraft / Stanford Cars training | 🤔 TBD | **Decided with the team after CUB results** |
| Prototype visualization | 📋 Planned | STEP 8 |
| Vogonet steel scrap transfer | 📋 Planned | STEP 10 (Phase 2, after dataset arrives) |

> This README follows the **Reproducibility Policy** (see §6). Only commands that actually work *now* are written in imperative form. Future stages are explicitly marked "📋 Planned" or "🤔 TBD".

---

## 🗺️ Pipeline

```text
ImageNet Pretraining (DeiT-S/16, provided by timm)
        ↓
CUB-200 Fine-tuning (200 classes)        ← STEP 6: smoke test
        ↓
[Next steps decided with the team after CUB results]
        ↓
Vogonet Steel Scrap (Phase 2, after dataset arrives this week of 2026-05-18)
```

---

## 1. Setup & Installation (📋 to be finalized in STEP 5)

> `pyproject.toml` is not defined yet. Once dependencies are pinned in STEP 5, this section will be replaced with the exact, working commands.

Planned form (for reference, not yet runnable):
```zsh
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv
uv sync                                            # sync dependencies
```

### Hardware
- **Development/verification machine**: NVIDIA GB10 (DGX Spark, aarch64, CUDA 13.0)
- **PyTorch wheel**: `torch==2.10.0+cu130` from `https://download.pytorch.org/whl/cu130`

### Datasets
- **CUB-200-2011**: [Download](https://www.vision.caltech.edu/datasets/cub-200-2011/)
- *FGVC-Aircraft, Stanford Cars: conditional, pending CUB results*

---

## 2. Training (📋 to be finalized in STEP 6)

After the CUB training command is validated end-to-end, this section will be filled with **the exact command, seed, and environment** so that any teammate can reproduce the same result.

Items to be recorded:
- Training command (e.g., `uv run python main.py --seed 42 ...`)
- Logs and checkpoint paths
- **Achieved accuracy (Top-1 %)** — for reproducibility verification

---

## 3. Evaluation (📋 after STEP 6)

Will be filled with the evaluation command and result numbers after CUB training completes.

---

## 4. Prototype Visualization ★ (📋 STEP 8, core deliverable)

This is the **core deliverable** of this repo — making the model's reasoning auditable for safety-critical industrial inspection.

Planned outputs:
- Per-class prototype gallery (source training image with patch-location overlay)
- Top-K activated prototypes per test image
- Side-by-side "this patch looks like that prototype" visualization

Use case:
> "Why was this scrap classified as DANGEROUS?"
> → "Patch X matches prototype P_47 (gas-cylinder valve pattern)."

---

## 5. 🔗 Related Repos (Forge-AI-Core)

| Repo | Paradigm | Backbone | Lead |
| :--- | :--- | :--- | :--- |
| [TransFG-Study](https://github.com/Forge-AI-Core/TransFG-Study) | Post-hoc PSM | ViT-B/16 | myself |
| [FGVC-PIM](https://github.com/Forge-AI-Core/FGVC-PIM) | Plug-in Module | Swin-T | team lead |
| [FGVC-RA-CNN](https://github.com/Forge-AI-Core/FGVC-RA-CNN) | Recurrent Attention | VGG/ResNet | team |
| **ProtoPFormer-Study** | **Intrinsic Prototype** | **DeiT-S/16** | **myself** |

---

## 6. Reproducibility Policy

This repo follows the team policy: **any teammate should be able to read this README, run the committed files as-is, and obtain the same result.**

- Every training command must use a **fixed seed** (`--seed 42`)
- Dependencies are pinned via `uv.lock`
- Training results (accuracy, etc.) are recorded as numbers in §2 / §3
- Stages that are not yet runnable are marked "📋 Planned" or "🤔 TBD" — no aspirational commands left in the README

---

## Citation

```bibtex
@inproceedings{xue2022protopformer,
  title={ProtoPFormer: Concentrating on Prototypical Parts in Vision Transformers for Interpretable Image Recognition},
  author={Xue, Mengqi and Huang, Qihan and Zhang, Haofei and Cheng, Lechao and Song, Jie and Wu, Minghui and Song, Mingli},
  booktitle={ECCV},
  year={2022}
}
```

---

### Acknowledgement
- Official ProtoPFormer implementation: [zju-vipa/ProtoPFormer](https://github.com/zju-vipa/ProtoPFormer)
- Team fine-grained research track: [Forge-AI-Core](https://github.com/Forge-AI-Core)
- AIFFEL Modulabs × Vogonet Co., Ltd. industry-academia project
