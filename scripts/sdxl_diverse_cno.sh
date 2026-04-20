#!/bin/bash

# ==========================================
# CNO (InfoNCE) MS-COCO Generation Script
# ==========================================

gpu=0
model="sdxl"   # "sd15" 또는 "sdxl" 선택
cfg_w=6.0
NFE=50         

num_samples=2000
b_size=5

seed=42

# CNO InfoNCE Parameters
temp=0.1
window_size=16
gamma=1.0
N=3
lr=0.01

save_dir="results_${model}"
prompt_dir="examples/assets/coco_v1.txt"

CUDA_VISIBLE_DEVICES=$gpu python text_to_mscoco.py \
    --model "$model" \
    --NFE $NFE \
    --cfg_guidance $cfg_w \
    --workdir "$save_dir" \
    --prompt_dir "$prompt_dir" \
    --b_size $b_size \
    --num_samples $num_samples \
    --seed $seed \
    --use_cno \
    --i_opt_lr $lr \
    --i_opt_iter $N \
    --gamma $gamma \
    --window_size $window_size \
    --infoNCE_temp $temp
