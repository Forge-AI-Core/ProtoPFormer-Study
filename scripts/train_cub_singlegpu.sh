#!/bin/bash
# CUB-200 단일 GPU 학습 (NVIDIA GB10) — torch.distributed.launch 미사용.
# main.py 의 init_distributed_mode()가 분산 환경변수 부재를 감지해 단일 GPU 모드로 자동 동작.
#
# 사용법 (ProtoPFormer 루트에서):
#   bash scripts/train_cub_singlegpu.sh [batch_size] [epochs]
#   예) smoke test:  bash scripts/train_cub_singlegpu.sh 64 3
#       본 학습:     bash scripts/train_cub_singlegpu.sh 128 200
#
# 전제: proto_venv 활성화, data → ../AiffelThon01/data 링크에 CUB_200_2011 존재.

export PYTHONPATH=./:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export MPLBACKEND=Agg   # 화면 없는 렌더링 backend → matplotlib의 tkinter(GUI) 의존성 회피

model=deit_small_patch16_224
batch_size=${1:-64}
epochs=${2:-3}
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
data_set=CUB2011U
prototype_num=2000
data_path=data        # ProtoPFormer/data → ../AiffelThon01/data (CUB_200_2011 포함)
output_dir=output/cub_singlegpu

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
    --ppc_mean_coe=$ppc_mean_coe
