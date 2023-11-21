DEVICE=0
dataset=allenai/real-toxicity-prompts
batch_size=32
pruning_number=$1
seed=42
model_name_or_path=${PWD}/models/distilgpt2_finetune


CUDA_VISIBLE_DEVICES=$DEVICE python3 real-toxicity_test.py -dataset $dataset -method imp_pruning -model_name_or_path $model_name_or_path -seed $seed -b $batch_size -pruning_number $pruning_number
