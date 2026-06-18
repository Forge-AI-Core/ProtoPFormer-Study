"""
ProtoPFormer Bogonet test set inference + threshold optimization + PR curve.

Workflow (per crop):
1. Load best checkpoint for given crop (25pct 또는 0pct)
2. Inference on val → softmax probabilities + labels
3. Sweep τ on val (rule: `P(danger) ≥ τ → danger; else argmax(cut, excluded)`)
   → balanced_acc 최대 τ* 결정
4. Inference on test set
5. Compute argmax + τ*-applied metrics, per-class PR curves, confusion matrix
6. Save artifacts to `result/<crop>/test/`

Usage:
    proto_venv/bin/python test_bogonet.py --expansion 25pct
    proto_venv/bin/python test_bogonet.py --expansion 0pct

또는 uv 환경이라면:
    python test_bogonet.py --expansion 25pct
"""

import os
import sys
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

# ProtoPFormer 가 BOGONET_EXPANSION 환경변수로 crop 선택 (tools/datasets.py:Bogonet)
# → CLI 받은 expansion 으로 미리 설정해두고 import.
_pre_args = argparse.ArgumentParser(add_help=False)
_pre_args.add_argument("--expansion", required=True, choices=["0pct", "25pct"])
_known, _ = _pre_args.parse_known_args()
os.environ["BOGONET_EXPANSION"] = _known.expansion

import torch
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)
from torch.utils.data import DataLoader, SequentialSampler
from torchvision import transforms
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import protopformer
from tools.datasets import Bogonet
from tools.preprocess import mean, std

CLASS_NAMES = ["cut", "danger", "excluded"]
DANGER_IDX = CLASS_NAMES.index("danger")  # = 1


# ─── CLI (full) ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--expansion",
    required=True,
    choices=["0pct", "25pct"],
    help="Test set crop expansion. testset 에 10pct 는 없음.",
)
parser.add_argument(
    "--checkpoint",
    default=None,
    help="Best checkpoint path. 기본: output/bogonet[_<crop>]/.../checkpoints/epoch-best.pth",
)
parser.add_argument(
    "--val-root",
    default=None,
    help="Val data root (classification_split). 기본: Bogonet/Bogonet_data/split02/classification_split",
)
parser.add_argument(
    "--test-root",
    default=None,
    help="Test data root (testset). 기본: AiffelThon01/Bogonet/Bogonet_data/testset",
)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--input-size", type=int, default=224)
parser.add_argument(
    "--output-dir",
    default=None,
    help="결과 저장 위치. 기본: result/<expansion>/test",
)

# ProtoPFormer model 구조 (학습 때와 동일해야 strict load)
parser.add_argument("--base-architecture", default="deit_small_patch16_224")
parser.add_argument("--prototype-shape", nargs="+", type=int, default=[30, 192, 1, 1])
parser.add_argument("--prototype-activation-function", default="log")
parser.add_argument("--add-on-layers-type", default="regular")
parser.add_argument("--reserve-layers", nargs="+", type=int, default=[11])
parser.add_argument("--reserve-token-nums", nargs="+", type=int, default=[81])
parser.add_argument("--use-global", action="store_true", default=True)
parser.add_argument("--no-use-global", dest="use_global", action="store_false")
parser.add_argument("--use-ppc-loss", action="store_true", default=False)
parser.add_argument("--ppc-cov-thresh", type=float, default=1.0)
parser.add_argument("--ppc-mean-thresh", type=float, default=2.0)
parser.add_argument("--global-coe", type=float, default=0.5)
parser.add_argument("--global-proto-per-class", type=int, default=10)

args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
exp = args.expansion


# ─── Paths ──────────────────────────────────────────────────────────────────────
def resolve_checkpoint() -> Path:
    if args.checkpoint:
        return Path(args.checkpoint).expanduser()

    # ProtoPFormer 학습 결과는 output/bogonet[_<crop>]/Bogonet/<arch>/<run_name>/checkpoints/epoch-best.pth
    base = PROJECT_ROOT / ("output/bogonet" if exp == "25pct" else f"output/bogonet_{exp}") / "Bogonet"
    if not base.is_dir():
        raise FileNotFoundError(
            f"학습 출력 디렉토리 없음: {base}\n--checkpoint 로 직접 지정 가능."
        )

    # 가장 epoch 많은 run 선택 (200 > 150 > 15 > 3 ...)
    candidates = sorted(
        base.glob(f"{args.base_architecture}/*-{args.prototype_activation_function}-*-*-protopformer/checkpoints/epoch-best.pth"),
        key=lambda p: int(p.parent.parent.name.split("-")[3]) if "-" in p.parent.parent.name else 0,
        reverse=True,
    )
    if not candidates:
        # 더 자유로운 패턴으로 재시도
        candidates = sorted(base.glob("**/checkpoints/epoch-best.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"epoch-best.pth 못 찾음 under {base}")
    return candidates[0]


ckpt_path = resolve_checkpoint()
val_root = Path(args.val_root) if args.val_root else PROJECT_ROOT / "data/Bogonet_data/split02/classification_split"
if not val_root.is_dir():
    # 다른 위치 시도
    alt = Path("/home/changilkim/Documents/aiffel_class/AiffelThon01/Bogonet/Bogonet_data/split02/classification_split")
    if alt.is_dir():
        val_root = alt
test_root = Path(args.test_root) if args.test_root else Path(
    "/home/changilkim/Documents/aiffel_class/AiffelThon01/Bogonet/Bogonet_data/testset"
)

assert val_root.is_dir(), f"Val root 없음: {val_root}"
assert test_root.is_dir(), f"Test root 없음: {test_root}"

output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / f"result/{exp}/test"
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Expansion : {exp}")
print(f"Device    : {DEVICE}")
print(f"Ckpt      : {ckpt_path}")
print(f"Val root  : {val_root}")
print(f"Test root : {test_root}")
print(f"Output    : {output_dir}\n")


# ─── Transforms (eval = ImageNet mean/std, Resize→CenterCrop) ──────────────────
def eval_transform(input_size: int):
    # timm create_transform(is_training=False) 등가 — 기본 crop_pct=0.875
    crop_pct = 0.875
    scale_size = int(input_size / crop_pct)
    return transforms.Compose(
        [
            transforms.Resize(scale_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


eval_tf = eval_transform(args.input_size)


# ─── Datasets ───────────────────────────────────────────────────────────────────
# Bogonet 은 train=False 면 val split 을 본다 (env var BOGONET_EXPANSION 기준)
valset = Bogonet(root=str(val_root), train=False, transform=eval_tf)
print(f"Val   : {len(valset)}  (class_counts={list(Counter([y for _, y in valset.samples]).items())})")


class BogonetTestDataset(torch.utils.data.Dataset):
    """testset 구조: <root>/crops_<expansion>/<class>/*.jpg (train/val 분할 없음)."""

    _EXTS = (".jpg", ".jpeg", ".png")

    def __init__(self, root, expansion, transform):
        cls_root = Path(root) / f"crops_{expansion}"
        self.transform = transform
        self.samples = []
        for label, cls in enumerate(CLASS_NAMES):
            cls_dir = cls_root / cls
            assert cls_dir.is_dir(), f"클래스 디렉토리 없음: {cls_dir}"
            for fp in sorted(cls_dir.iterdir()):
                if fp.suffix.lower() in self._EXTS:
                    self.samples.append((fp, label))
        self.labels = [lb for _, lb in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def class_counts(self):
        c = Counter(self.labels)
        return [c[i] for i in range(len(CLASS_NAMES))]


testset = BogonetTestDataset(root=str(test_root), expansion=exp, transform=eval_tf)
print(f"Test  : {len(testset)}  (class_counts={testset.class_counts()})\n")

val_loader = DataLoader(
    valset,
    sampler=SequentialSampler(valset),
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=True,
)
test_loader = DataLoader(
    testset,
    sampler=SequentialSampler(testset),
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=True,
)


# ─── Model build + checkpoint load ─────────────────────────────────────────────
model = protopformer.construct_PPNet(
    base_architecture=args.base_architecture,
    pretrained=False,  # weight 는 epoch-best.pth 에서 통째로
    img_size=args.input_size,
    prototype_shape=args.prototype_shape,
    num_classes=3,
    reserve_layers=args.reserve_layers,
    reserve_token_nums=args.reserve_token_nums,
    use_global=args.use_global,
    use_ppc_loss=args.use_ppc_loss,
    ppc_cov_thresh=args.ppc_cov_thresh,
    ppc_mean_thresh=args.ppc_mean_thresh,
    global_coe=args.global_coe,
    global_proto_per_class=args.global_proto_per_class,
    prototype_activation_function=args.prototype_activation_function,
    add_on_layers_type=args.add_on_layers_type,
)

ck = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
missing, unexpected = model.load_state_dict(sd, strict=False)
if missing:
    print(f"⚠ missing keys: {len(missing)} (예: {missing[:3]})")
if unexpected:
    print(f"⚠ unexpected keys: {len(unexpected)} (예: {unexpected[:3]})")
model = model.to(DEVICE).eval()
print(f"Loaded checkpoint  (best={ck.get('best_acc', ck.get('best_balanced', 'n/a'))})\n")


# ─── Inference helpers ────────────────────────────────────────────────────────
@torch.no_grad()
def get_probs_labels(loader, desc="Inference"):
    all_probs, all_labels = [], []
    for x, y in tqdm(loader, desc=desc, leave=False):
        x = x.to(DEVICE)
        out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def predict_with_threshold(probs, tau):
    """Rule: P(danger) ≥ τ → danger; else argmax(cut, excluded)."""
    danger_p = probs[:, DANGER_IDX]
    other = probs[:, [0, 2]]
    other_pred = np.where(other.argmax(axis=1) == 0, 0, 2)
    return np.where(danger_p >= tau, DANGER_IDX, other_pred)


# ─── Val inference + threshold sweep ─────────────────────────────────────────
print(">> Val inference (threshold 결정용)")
val_probs, val_labels = get_probs_labels(val_loader, desc="Val")

print(">> Threshold sweep on val")
taus = np.arange(0.05, 0.96, 0.05)
sweep = []
for tau in taus:
    pred = predict_with_threshold(val_probs, tau)
    acc = float((pred == val_labels).mean())
    p_d = precision_score(val_labels == DANGER_IDX, pred == DANGER_IDX, zero_division=0)
    r_d = recall_score(val_labels == DANGER_IDX, pred == DANGER_IDX, zero_division=0)
    f1_d = f1_score(val_labels == DANGER_IDX, pred == DANGER_IDX, zero_division=0)
    recalls = [float((pred[val_labels == c] == c).mean()) for c in [0, 1, 2]]
    bacc = float(np.mean(recalls))
    beta = 0.5
    fbeta = (
        (1 + beta**2) * p_d * r_d / (beta**2 * p_d + r_d)
        if (beta**2 * p_d + r_d) > 0
        else 0.0
    )
    sweep.append(
        {
            "tau": float(tau),
            "acc": acc,
            "bacc": bacc,
            "danger_p": float(p_d),
            "danger_r": float(r_d),
            "danger_f1": float(f1_d),
            "danger_fbeta05": float(fbeta),
        }
    )

best_bacc_row = max(sweep, key=lambda r: r["bacc"])
best_fbeta_row = max(sweep, key=lambda r: r["danger_fbeta05"])
TAU_STAR = best_bacc_row["tau"]

print(f"τ* (val balanced_acc 최대) = {TAU_STAR:.2f}  bacc={best_bacc_row['bacc']:.4f}")
print(f"(참고) F-beta(0.5) 최대 τ  = {best_fbeta_row['tau']:.2f}  fbeta={best_fbeta_row['danger_fbeta05']:.4f}")


# ─── Test inference ─────────────────────────────────────────────────────────
print(f"\n>> Test inference (testset/crops_{exp})")
test_probs, test_labels = get_probs_labels(test_loader, desc="Test")

test_pred_argmax = test_probs.argmax(axis=1)
test_pred_tau = predict_with_threshold(test_probs, TAU_STAR)


# ─── Metrics ────────────────────────────────────────────────────────────────
def compute_metrics(labels, preds, class_names):
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    acc = float(cm.trace() / cm.sum()) if cm.sum() else 0.0
    per_class = []
    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class.append(
            {
                "name": cls,
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "support": int(cm[i, :].sum()),
            }
        )
    bacc = float(np.mean([c["recall"] for c in per_class]))
    return cm, acc, bacc, per_class


cm_argmax, acc_argmax, bacc_argmax, perc_argmax = compute_metrics(
    test_labels, test_pred_argmax, CLASS_NAMES
)
cm_tau, acc_tau, bacc_tau, perc_tau = compute_metrics(
    test_labels, test_pred_tau, CLASS_NAMES
)


# ─── PR curves (one-vs-rest) ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ap_dict = {}
for c, cls in enumerate(CLASS_NAMES):
    y_true = (test_labels == c).astype(int)
    y_score = test_probs[:, c]
    p_arr, r_arr, _ = precision_recall_curve(y_true, y_score)
    ap = float(average_precision_score(y_true, y_score))
    ap_dict[cls] = ap
    ax.plot(r_arr, p_arr, label=f"{cls}  AP={ap:.3f}")
mean_ap = float(np.mean(list(ap_dict.values())))
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title(f"PR curves (test, {exp})  mAP={mean_ap:.3f}")
ax.legend(loc="lower left")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(output_dir / "pr_curve.png", dpi=120)
plt.close()


# ─── Confusion matrix figure ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax_, cm_, title in zip(
    axes, [cm_argmax, cm_tau], ["argmax", f"τ={TAU_STAR:.2f}"]
):
    ax_.imshow(cm_, cmap="Blues")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax_.text(j, i, str(cm_[i, j]), ha="center", va="center")
    ax_.set_xticks(range(len(CLASS_NAMES)))
    ax_.set_xticklabels(CLASS_NAMES)
    ax_.set_yticks(range(len(CLASS_NAMES)))
    ax_.set_yticklabels(CLASS_NAMES)
    ax_.set_xlabel("Predicted")
    ax_.set_ylabel("True")
    ax_.set_title(f"Confusion ({title})  acc={(cm_.trace() / cm_.sum()) * 100:.1f}%")
plt.tight_layout()
plt.savefig(output_dir / "confusion_matrix.png", dpi=120)
plt.close()


# ─── Save markdown reports ────────────────────────────────────────────────
def fmt_per_class(perc):
    rows = []
    for c in perc:
        rows.append(
            f"| {c['name']:8s} | {c['precision'] * 100:6.2f} | "
            f"{c['recall'] * 100:6.2f} | {c['f1'] * 100:6.2f} | {c['support']:5d} |"
        )
    return "\n".join(rows)


metrics_md = f"""# Test Eval — ProtoPFormer-Study {exp} crop

- Checkpoint: `{ckpt_path}`
- Val set    : split02 val ({len(valset)} 장)
- Test set   : `testset/crops_{exp}` ({len(testset)} 장)
- Test class counts (cut/danger/excluded): {testset.class_counts()}  (※ danger 압도)

## Validation 에서 결정한 τ*

- **τ\\* = {TAU_STAR:.2f}** (val balanced_acc 최대)
- 규칙: `P(danger) ≥ τ → danger; 아니면 argmax({{cut, excluded}})`
- Val balanced_acc(τ\\*) = {best_bacc_row['bacc']:.4f}
- 참고 — F-beta(0.5) 최대 τ = {best_fbeta_row['tau']:.2f}, fbeta={best_fbeta_row['danger_fbeta05']:.4f}

(threshold sweep 전체: [`threshold_sweep.md`](threshold_sweep.md))

## Test 결과 — A. argmax

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
{fmt_per_class(perc_argmax)}

- Accuracy = **{acc_argmax * 100:.2f}%**
- Balanced acc = **{bacc_argmax * 100:.2f}%**

### Confusion matrix (argmax)
```
              cut  danger  excluded
""" + "\n".join(
    f"  {CLASS_NAMES[i]:8s} {cm_argmax[i, 0]:5d}  {cm_argmax[i, 1]:5d}    {cm_argmax[i, 2]:5d}"
    for i in range(len(CLASS_NAMES))
) + f"""
```

## Test 결과 — B. τ\\*={TAU_STAR:.2f} 적용

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
{fmt_per_class(perc_tau)}

- Accuracy = **{acc_tau * 100:.2f}%**
- Balanced acc = **{bacc_tau * 100:.2f}%**

### Confusion matrix (τ\\* 적용)
```
              cut  danger  excluded
""" + "\n".join(
    f"  {CLASS_NAMES[i]:8s} {cm_tau[i, 0]:5d}  {cm_tau[i, 1]:5d}    {cm_tau[i, 2]:5d}"
    for i in range(len(CLASS_NAMES))
) + f"""
```

## PR Curve (one-vs-rest, test)

| class | AP |
|---|---:|
| cut | {ap_dict['cut']:.4f} |
| danger | {ap_dict['danger']:.4f} |
| excluded | {ap_dict['excluded']:.4f} |
| **mean AP** | **{mean_ap:.4f}** |

![PR curves](pr_curve.png)

## Confusion Matrix

![Confusion matrix](confusion_matrix.png)
"""

(output_dir / "metrics.md").write_text(metrics_md, encoding="utf-8")

sweep_md_lines = [
    "# Threshold sweep on val",
    "",
    "규칙: `P(danger) ≥ τ → danger; 아니면 argmax({cut, excluded})`. 학습된 모델 그대로 (재학습 0).",
    "",
    "| τ | acc | bacc | danger P | danger R | danger F1 | danger F-beta(0.5) |",
    "|---:|---:|---:|---:|---:|---:|---:|",
]
for r in sweep:
    marker = " **★**" if abs(r["tau"] - TAU_STAR) < 0.001 else ""
    sweep_md_lines.append(
        f"| {r['tau']:.2f}{marker} | {r['acc']:.4f} | {r['bacc']:.4f} | "
        f"{r['danger_p']:.4f} | {r['danger_r']:.4f} | {r['danger_f1']:.4f} | "
        f"{r['danger_fbeta05']:.4f} |"
    )
sweep_md_lines.append("")
sweep_md_lines.append(f"**선택된 τ\\* = {TAU_STAR:.2f}** (val balanced_acc 최대)")
(output_dir / "threshold_sweep.md").write_text("\n".join(sweep_md_lines), encoding="utf-8")


# ─── Summary stdout ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"✓ 결과 저장 완료 → {output_dir}")
print(f"    metrics.md  threshold_sweep.md  pr_curve.png  confusion_matrix.png")
print("=" * 60)
print(f"\n>>> Test {exp} summary")
print(f"  Test class counts (cut/danger/excluded): {testset.class_counts()}")
print(f"  argmax        : acc {acc_argmax * 100:6.2f}% / bacc {bacc_argmax * 100:6.2f}%")
print(f"  τ*={TAU_STAR:.2f} 적용 : acc {acc_tau * 100:6.2f}% / bacc {bacc_tau * 100:6.2f}%")
print(f"  PR-AUC (mAP)  : {mean_ap:.4f}")
print(
    f"  per-class AP  : cut {ap_dict['cut']:.3f}, danger {ap_dict['danger']:.3f}, excluded {ap_dict['excluded']:.3f}"
)
