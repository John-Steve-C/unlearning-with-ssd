# this is my unlearning test
# This is the main file to run all experiments
# Bash file to run different seeds (corresponding to the value) across all tasks
# Pass the GPU ID with the first parameter (e.g., 0; check via nvidia-smi)

reset_cuda(){
    sleep 10    
}

DEVICE=0
seed=42

#############################################################
################ RANDOM FORGETTING ##################
#############################################################
# export WANDB_RUN_NAME="unlearn-toxigen-distilgpt2"
# export DISTILGPT2_PATH=./models/distilgpt2
export CLASSIFIER_PATH=./models/roberta_toxicity_classifier


forget_perc=0.1 # forgetting propotion
dataset=allenai/real-toxicity-prompts
origin_model=distilgpt2   #princeton-nlp/Sheared-LLaMA-1.3B #
batch_size=16
n_classes=2
pruning_percent=0.2

# TODO: need to modify this!
# model_name_or_path=./models/
model_name_or_path=./models/distilgpt2_finetune

# Run the Python script
# CUDA_VISIBLE_DEVICES=$DEVICE python3 toxic_gen_test.py -model $model -dataset $dataset -classes $n_classes -method baseline -forget_perc $forget_perc -model_path $model_path -seed $seed -b $batch_size
# reset_cuda
# CUDA_VISIBLE_DEVICES=$DEVICE python3 toxic_gen_test.py -model $model -dataset $dataset -classes $n_classes -method finetune -forget_perc $forget_perc -model_path $model_path -seed $seed -b $batch_size
# reset_cuda

# CUDA_VISIBLE_DEVICES=$DEVICE python3 real-toxicity_test.py -origin_model $origin_model -dataset $dataset -classes $n_classes -method baseline -forget_perc $forget_perc -model_name_or_path $model_name_or_path -seed $seed -b $batch_size
# reset_cuda
# CUDA_VISIBLE_DEVICES=$DEVICE python3 real-toxicity_test.py -origin_model $origin_model -dataset $dataset -classes $n_classes -method finetune -forget_perc $forget_perc -model_name_or_path $model_name_or_path -seed $seed -b $batch_size
# reset_cuda
# CUDA_VISIBLE_DEVICES=$DEVICE python3 real-toxicity_test.py -origin_model $origin_model -dataset $dataset -classes $n_classes -method pdr_tuning -forget_perc $forget_perc -model_name_or_path $model_name_or_path -seed $seed -b $batch_size
# # reset_cuda

# origin = abs + zero

# test baseline

# CUDA_VISIBLE_DEVICES=$DEVICE python3 real-toxicity_test.py -origin_model $origin_model -dataset $dataset -classes $n_classes -forget_perc $forget_perc -model_name_or_path ./models/distilgpt2_baseline -seed $seed -b $batch_size -pruning_percent $pruning_percent \
#     -method finetune \
#     -forget_type abs \
#     -modify_method zero \
#     -forget_importances_pkl None1 \
#     -retain_importances_pkl None2 \
#     > info_zero.txt
#     # -forget_importances_pkl imp_forget_baseline \
#     # -retain_importances_pkl imp_retain_baseline \

# reset_cuda

# CUDA_VISIBLE_DEVICES=$DEVICE python3 real-toxicity_test.py -origin_model $origin_model -dataset $dataset -classes $n_classes -forget_perc $forget_perc -model_name_or_path $model_name_or_path -seed $seed -b $batch_size -pruning_percent $pruning_percent \
#     -method imp_pruning_large \
#     -forget_type perturb \
#     -modify_method zero \
#     -load_from_file False \
#     -use_sample \
#     > info_zero.txt

# reset_cuda

# CUDA_VISIBLE_DEVICES=$DEVICE python3 real-toxicity_test.py -origin_model $origin_model -dataset $dataset -classes $n_classes -forget_perc $forget_perc -model_name_or_path $model_name_or_path -seed $seed -b $batch_size -pruning_percent $pruning_percent \
#     -method reverse_gradient \
#     -forget_importances_pkl imp_forget_reverse_test \
#     -retain_importances_pkl imp_retain_reverse_test \
#     > info_test_2.txt

# reset_cuda

# CUDA_VISIBLE_DEVICES=$DEVICE python3 real-toxicity_test.py -origin_model $origin_model -dataset $dataset -classes $n_classes -forget_perc $forget_perc -model_name_or_path $model_name_or_path -seed $seed -b $batch_size -pruning_percent $pruning_percent \
#     -method imp_pruning \
#     -forget_type abs \
#     -modify_method reverse \
#     -pruning_percent 0.5 \
#     -forget_importances_pkl imp_forget_zero_tmp \
#     -retain_importances_pkl imp_retain_zero_tmp \
#     -load_from_file  \
#     > info_zero.txt

# reset_cuda

# CUDA_VISIBLE_DEVICES=$DEVICE python3 real-toxicity_test.py -origin_model $origin_model -dataset $dataset -classes $n_classes -forget_perc $forget_perc -model_name_or_path $model_name_or_path -seed $seed -b $batch_size -pruning_percent $pruning_percent \
#     -method mixture_pruning \
#     -forget_type std \
#     -modify_method zero \
#     -pruning_percent 0.2 \
#     -pruning_percent_2 0.5 \
#     -forget_importances_pkl imp_forget_tmp_std \
#     -retain_importances_pkl imp_retain_tmp_std \
#     > info_perturb_2.txt

CUDA_VISIBLE_DEVICES=$DEVICE python3 real-toxicity_test.py -origin_model $origin_model -dataset $dataset -classes $n_classes -forget_perc $forget_perc -model_name_or_path $model_name_or_path -seed $seed -b $batch_size \
    -method mixture_pruning \
    -forget_type abs \
    -modify_method zero \
    -pruning_percent 0.2 \
    -forget_importances_pkl forget_imp_3 \
    -forget_importances_pkl_2 forget_imp_4 \
    -load_from_file \
    > info_mix.txt

reset_cuda

# CUDA_VISIBLE_DEVICES=$DEVICE python3 real-toxicity_test.py -origin_model $origin_model -dataset $dataset -classes $n_classes -forget_perc $forget_perc -model_name_or_path $model_name_or_path -seed $seed -b $batch_size -pruning_percent $pruning_percent \
#     -method mixture_pruning \
#     -forget_type abs \
#     -modify_method zero \
#     -pruning_percent 0.8 \
#     -forget_importances_pkl imp_forget_perturb_large \
#     -retain_importances_pkl imp_retain_perturb_large \
#     -load_from_file True \
#     > info_perturb_3.txt

# reset_cuda

# CUDA_VISIBLE_DEVICES=$DEVICE python3 real-toxicity_test.py -origin_model $origin_model -dataset $dataset -classes $n_classes -forget_perc $forget_perc -model_name_or_path $model_name_or_path -seed $seed -b $batch_size -pruning_percent $pruning_percent \
#     -method imp_pruning \
#     -forget_type perturb \
#     -modify_method zero \
#     -forget_importances_pkl imp_forget_perturb_small \
#     -retain_importances_pkl imp_retain_perturb_small \
#     > info_perturb_small.txt

# reset_cuda

# CUDA_VISIBLE_DEVICES=$DEVICE python3 real-toxicity_test.py -origin_model $origin_model -dataset $dataset -classes $n_classes -forget_perc $forget_perc -model_name_or_path $model_name_or_path -seed $seed -b $batch_size -pruning_percent $pruning_percent \
#     -method imp_pruning_large \
#     -forget_type grad \
#     -modify_method zero \
#     -forget_importances_pkl imp_forget_grad \
#     -retain_importances_pkl imp_retain_grad \
#     -load_from_file True \
#     > info_grad.txt

# reset_cuda

# CUDA_VISIBLE_DEVICES=$DEVICE python3 real-toxicity_test.py -origin_model $origin_model -dataset $dataset -classes $n_classes -forget_perc $forget_perc -model_name_or_path $model_name_or_path -seed $seed -b $batch_size -pruning_percent $pruning_percent \
#     -method imp_pruning_large \
#     -forget_type grad \
#     -modify_method reverse \
#     -forget_importances_pkl imp_forget_grad \
#     -retain_importances_pkl imp_retain_grad \
#     -load_from_file True \
#     > info_grad_reverse.txt

# reset_cuda

# CUDA_VISIBLE_DEVICES=$DEVICE python3 real-toxicity_test.py -origin_model $origin_model -dataset $dataset -classes $n_classes -method imp_pruning_large -forget_perc $forget_perc -model_name_or_path $model_name_or_path -seed $seed -b $batch_size -use_sample
# reset_cuda
