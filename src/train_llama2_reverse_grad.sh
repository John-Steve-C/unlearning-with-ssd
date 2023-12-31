pip uninstall transformer-engine

export WANDB_RUN_NAME=$1
torchrun --nproc_per_node=8 train_llama2_reverse_grad.py \
    --model_name_or_path [PATH_TO_Llama-2-7b-hf] \
    --data_path "skg/toxigen-data" \
    --bf16 True \
    --output_dir [PATH_TO_LOG] \ 
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --retain_only True \
    --deepspeed deepspeed.json \
    --forget_importances_pkl [PATH_TO_PKL] \
    --retain_importances_pkl [PATH_TO_PKL]
