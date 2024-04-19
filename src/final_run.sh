reset_cuda(){
    sleep 10    
}

pip uninstall transformer-engine

# get grad importance
export WANDB_RUN_NAME=$1
torchrun --nproc_per_node=gpu get_grad_importance.py \
    --model_name_or_path ./models/distilgpt2_finetune\
    --bf16 True \
    --output_dir ./imps\ 
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

reset_cuda

export CLASSIFIER_PATH=./models/roberta_toxicity_classifier

seed=42
forget_perc=0.1 # forgetting propotion
dataset=allenai/real-toxicity-prompts
origin_model=distilgpt2   #princeton-nlp/Sheared-LLaMA-1.3B #
batch_size=16
n_classes=2
pruning_percent=0.2

# TODO: need to modify this!
model_name_or_path=./models/distilgpt2_finetune

# get forward importance & final pruning
python3 real-toxicity_test.py -origin_model $origin_model -dataset $dataset -classes $n_classes -forget_perc $forget_perc -model_name_or_path $model_name_or_path -seed $seed -b $batch_size \
    -method mixture_pruning \
    -forget_type abs \
    -modify_method zero \
    -pruning_percent $pruning_percent \
    -forget_importances_pkl forward_importance.pkl \
    -forget_importances_pkl_2 grad_importance.pkl \
    > info_final.txt