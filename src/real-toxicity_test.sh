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

forget_perc=0.1 # forgetting propotion
dataset=allenai/real-toxicity-prompts
origin_model=distilgpt2   #princeton-nlp/Sheared-LLaMA-1.3B #
batch_size=32
n_classes=2

# TODO: need to modify this!
# model_name_or_path=./models/
model_name_or_path=distilgpt2

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
# reset_cuda
CUDA_VISIBLE_DEVICES=$DEVICE python3 real-toxicity_test.py -origin_model $origin_model -dataset $dataset -classes $n_classes -method imp_pruning -forget_perc $forget_perc -model_name_or_path $model_name_or_path -seed $seed -b $batch_size -use_sample
reset_cuda
#CUDA_VISIBLE_DEVICES=$DEVICE python3 real-toxicity_test.py -origin_model $origin_model -dataset $dataset -classes $n_classes -method imp_pruning -forget_perc $forget_perc -model_name_or_path $model_name_or_path -seed $seed -b $batch_size


# CUDA_VISIBLE_DEVICES=$DEVICE python forget_random_main.py -net ResNet18 -dataset $dataset -classes $n_classes -gpu -method finetune -forget_perc $forget_perc -weight_path $weight_path -seed $seed
# reset_cuda
# CUDA_VISIBLE_DEVICES=$DEVICE python forget_random_main.py -net ResNet18 -dataset $dataset -classes $n_classes -gpu -method amnesiac -forget_perc $forget_perc -weight_path $weight_path -seed $seed
# reset_cuda
# CUDA_VISIBLE_DEVICES=$DEVICE python forget_random_main.py -net ResNet18 -dataset $dataset -classes $n_classes -gpu -method blindspot -forget_perc $forget_perc -weight_path $weight_path -seed $seed
# reset_cuda
# CUDA_VISIBLE_DEVICES=$DEVICE python forget_random_main.py -net ResNet18 -dataset $dataset -classes $n_classes -gpu -method UNSIR -forget_perc $forget_perc -weight_path $weight_path -seed $seed
# reset_cuda
# CUDA_VISIBLE_DEVICES=$DEVICE python forget_random_main.py -net ResNet18 -dataset $dataset -classes $n_classes -gpu -method retrain -forget_perc $forget_perc -weight_path $weight_path -seed $seed
# reset_cuda
