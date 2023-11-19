
DEVICE=0
forget_perc=0.1 # forgetting propotion
dataset=allenai/real-toxicity-prompts
origin_model=distilgpt2   #princeton-nlp/Sheared-LLaMA-1.3B #
batch_size=64
n_classes=2
pruning_number=50
seed=42

origin_model=distilgpt2
model_name_or_path=distilgpt2


CUDA_VISIBLE_DEVICES=$DEVICE python3 real-toxicity_test.py -origin_model $origin_model -dataset $dataset -classes $n_classes -method imp_pruning -forget_perc $forget_perc -model_name_or_path $model_name_or_path -seed $seed -b $batch_size -pruning_number $pruning_number
