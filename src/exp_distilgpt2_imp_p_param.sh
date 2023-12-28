DEVICE=0
dataset=allenai/real-toxicity-prompts
batch_size=32
pruning_percent=$1
seed=42
model_name_or_path=${PWD}/models//distilgpt2_finetune/
forget_type="abs"


CUDA_VISIBLE_DEVICES=$DEVICE python3 real-toxicity_test.py -dataset $dataset -method imp_pruning -model_name_or_path $model_name_or_path -seed $seed -b $batch_size -pruning_percent $pruning_percent -forget_type $forget_type -use_sample -neuron_name "mlp.c_proj"
