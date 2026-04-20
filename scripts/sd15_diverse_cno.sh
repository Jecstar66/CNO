gpu=0
model="sd15"  
method="ddim"
cfg_w=6.0
NFE=50         

num_samples=2000
b_size=5
repeated_num_samples=$(( $num_samples * $b_size ))

seed=42

type="infoNCE"
temp=0.1
window_size=16
gamma=1.0
N=3
lr=0.01
save_dir="results"
prompt_dir="examples/assets/coco_v1.txt"


CUDA_VISIBLE_DEVICES=$gpu python -m examples.text_to_mscoco \
    --model "$model" --method "$method" --NFE $NFE --cfg_guidance $cfg_w \
    --workdir "$save_dir" --prompt_dir "$prompt_dir" --b_size $b_size --num_samples $num_samples --seed $seed \
    --repeat_prompt --iopt_diverse --i_opt_lr $lr --i_opt_iter $N --iopt_loss_type "$type"  \
    --gamma $gamma --window_size $window_size --infoNCE_temp $temp \

