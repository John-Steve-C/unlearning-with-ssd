pip3 install datasets
# pip3 install transformers==4.28.1


export WANDB_RUN_NAME="finetune-distilgpt2-with-toxigen"
torchrun --nproc_per_node=1 toxic_gen_train.py \
    --model_name_or_path "distilgpt2" \
    --data_path "skg/toxigen-data" \
    --bf16 False \
    --output_dir ./model/ \
    --num_train_epochs 3 \
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
    --logging_steps 10 \
    --tf32 False

# export WANDB_RUN_NAME="finetune-llama2-with-toxigen-dev"
# torchrun --nproc_per_node=8 train.py \
#     --model_name_or_path /workspace/github/models/Llama-2-7b-hf \
#     --data_path "skg/toxigen-data" \
#     --bf16 True \
#     --output_dir /workspace/github/models/${WANDB_RUN_NAME} \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "steps" \
#     --eval_steps 100 \
#     --save_strategy "steps" \
#     --save_steps 2000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --fsdp "full_shard auto_wrap" \
#     --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer" \
#     --tf32 True
