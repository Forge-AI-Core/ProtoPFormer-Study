#!/bin/bash
# Bogonet 위험물 3-class transfer 학습 (단일 GPU, NVIDIA GB10).
# train_cub_singlegpu.sh 와 동일 구조. 차이: data_set=Bogonet, prototype 30개(3-class×10), DeiT-S ImageNet init.
#
# 데이터: AiffelThon01 의 Bogonet_EDA.ipynb 가 만든 split(splits/{train,val}.txt)을 그대로 사용.
#         → TransFG 실험과 동일 val set, leakage 0. crop expansion=25pct(tools/datasets.py Bogonet.EXPANSION).
#
# 사용법 (ProtoPFormer 루트에서):
#   bash scripts/train_bogonet.sh [batch_size] [epochs] [expansion] [resume_ckpt]
#   예) 25pct 본학습: bash scripts/train_bogonet.sh 64 200 25pct
#       10pct 본학습: bash scripts/train_bogonet.sh 64 150 10pct
#       재개:         bash scripts/train_bogonet.sh 64 150 10pct <output/bogonet_10pct/.../checkpoint-latest.pth>
#
# 전제: proto_venv 활성화. CUB checkpoint warm-start 안 함(아직 학습된 CUB ProtoPFormer 없음)
#       → DeiT-S ImageNet pretrained backbone 에서 출발 (CUB smoke test와 동일 init).

export PYTHONPATH=./:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export MPLBACKEND=Agg   # 화면 없는 렌더링 backend

model=deit_small_patch16_224
batch_size=${1:-64}
epochs=${2:-3}
expansion=${3:-25pct}   # crop expansion: 0pct / 10pct / 25pct (ablation)
resume_ckpt=${4:-}      # (선택) 4번째 인자에 checkpoint 경로 주면 거기서 이어서 학습
export BOGONET_EXPANSION=$expansion   # tools/datasets.py Bogonet 가 읽음
seed=1028

# Learning Rate
warmup_lr=1e-4
warmup_epochs=5
features_lr=1e-4
add_on_layers_lr=3e-3
prototype_vectors_lr=3e-3

# Optimizer & Scheduler
opt=adamw
sched=cosine
decay_epochs=10
decay_rate=0.1
weight_decay=0.05
input_size=224

# ProtoPFormer
use_global=True
use_ppc_loss=True
last_reserve_num=81
global_coe=0.5
ppc_cov_thresh=1.
ppc_mean_thresh=2.
global_proto_per_class=10
ppc_cov_coe=0.1
ppc_mean_coe=0.5
dim=192
reserve_layer_idx=11

ft=protopformer
data_set=Bogonet
# 팀 공통 image-level split(split02). crops_*_split/{train,val}/{class}/ 구조. tools/datasets.py Bogonet 가 읽음.
data_path=/home/changilkim/Documents/aiffel_class/AiffelThon01/Bogonet/Bogonet_data/split02/classification_split
prototype_num=30      # 3-class × 10 per class (assert num_prototypes % num_classes == 0)
dim=192
output_dir=output/bogonet_${expansion}   # expansion별 분리 (10pct ↔ 25pct 안 섞임)

python main.py \
    --base_architecture=$model \
    --data_set=$data_set \
    --data_path=$data_path \
    --input_size=$input_size \
    --output_dir=$output_dir/$data_set/$model/$seed-$opt-$weight_decay-$epochs-$ft \
    --model=$model \
    --batch_size=$batch_size \
    --seed=$seed \
    --opt=$opt \
    --sched=$sched \
    --warmup-epochs=$warmup_epochs \
    --warmup-lr=$warmup_lr \
    --decay-epochs=$decay_epochs \
    --decay-rate=$decay_rate \
    --weight_decay=$weight_decay \
    --epochs=$epochs \
    --finetune=$ft \
    --features_lr=$features_lr \
    --add_on_layers_lr=$add_on_layers_lr \
    --prototype_vectors_lr=$prototype_vectors_lr \
    --prototype_shape $prototype_num $dim 1 1 \
    --reserve_layers $reserve_layer_idx \
    --reserve_token_nums $last_reserve_num \
    --use_global=$use_global \
    --use_ppc_loss=$use_ppc_loss \
    --ppc_cov_thresh=$ppc_cov_thresh \
    --ppc_mean_thresh=$ppc_mean_thresh \
    --global_coe=$global_coe \
    --global_proto_per_class=$global_proto_per_class \
    --ppc_cov_coe=$ppc_cov_coe \
    --ppc_mean_coe=$ppc_mean_coe \
    --balanced_sampler \
    ${resume_ckpt:+--resume=$resume_ckpt}
