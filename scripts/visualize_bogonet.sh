#!/bin/bash
# Bogonet prototype 시각화 — 학습된 ProtoPFormer가 어느 패치를 어느 prototype에 활성시키는지 히트맵 생성.
# 사용법 (ProtoPFormer 루트, proto_venv 활성화 상태):
#   bash scripts/visualize_bogonet.sh [modelfile]
#   예) bash scripts/visualize_bogonet.sh epoch-best.pth
#
# 산출물: output/bogonet_viz/Bogonet/deit_small_patch16_224-visualize-test/slim_gaussian/category_{0,1,2}/...

export PYTHONPATH=./:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export MPLBACKEND=Agg

model=deit_small_patch16_224
modelfile=${1:-epoch-best.pth}
data_set=Bogonet
data_path=/home/changilkim/Documents/aiffel_class/AiffelThon01/Bogonet/Bogonet_data/split02/classification_split
modeldir=output/bogonet/Bogonet/deit_small_patch16_224/1028-adamw-0.05-200-protopformer/checkpoints
output_dir=output/bogonet_viz

dim=192
prototype_num=30          # 3-class × 10 per class
global_proto_per_class=10
reserve_layer_idx=11
reserve_token_nums=81

python main_visualize.py \
    --finetune=visualize \
    --modeldir=$modeldir \
    --model=$modelfile \
    --data_set=$data_set \
    --data_path=$data_path \
    --prototype_shape $prototype_num $dim 1 1 \
    --reserve_layers=$reserve_layer_idx \
    --reserve_token_nums=$reserve_token_nums \
    --use_global=True \
    --use_ppc_loss=True \
    --global_coe=0.5 \
    --global_proto_per_class=$global_proto_per_class \
    --base_architecture=$model \
    --batch_size=60 \
    --visual_type=slim_gaussian \
    --output_dir=$output_dir \
    --use_gauss=True \
    --vis_classes 0 1 2
