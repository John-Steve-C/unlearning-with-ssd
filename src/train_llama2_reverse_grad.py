import datasets
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import argparse
import math
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer
import pickle
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM
from transformers import AdamW
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="distilgpt2")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    retain_only: Optional[bool] = field(default=False)
    forget_importances_pkl: str = field(default="None")
    retain_importances_pkl: str = field(default="None")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    pruning_percent: float = field(default=0.1)
    


import torch
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(device)


class CustomTrainer(Trainer):
    def __init__(self, *args, mask,  **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = mask

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        selected_named_parameters = [(name, param) for name, param in self.model.named_parameters() if "mlp.down_proj" in name]
        selected_named_parameters = sorted(selected_named_parameters, key=lambda x: x[0])
        self.optimizer=CustomOpt([{'params': param, 'name': name} for name, param in selected_named_parameters], lr=5e-5, mask=self.mask)        
        self.lr_scheduler=transformers.get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                                        num_warmup_steps=2000,
                                                                        num_training_steps=num_training_steps)

def get_mask(
        model,
        score: list,
        pruning_percent: float,
    ) -> torch.Tensor:

    neuron_name = "mlp.down_proj"
    layers = 0
    for (name, module) in model.named_modules():
        if neuron_name in name: #and module.__class__ == transformers.pytorch_utils.Conv1D:
            print(name) # model.layers.27.mlp.down_proj
            layers += 1
            weight_shape = module.weight.shape
    layers = layers
    neuron_number_per_layer = weight_shape[1]
    total_neuron_number = layers * weight_shape[1]

    # only important neurons will be set to 1 for updating
    mask = torch.zeros(layers, neuron_number_per_layer)
    pruning_number = int(total_neuron_number * pruning_percent)

    score_pair = [(s, id) for id, s in enumerate(score)]
    score_pair.sort(key=lambda x: x[0], reverse=True)   # true means descending

    for i in tqdm(range(pruning_number)):
        del_pair = score_pair[i]
        id = del_pair[1]
        neuron_id = id % neuron_number_per_layer
        layer_id = id // neuron_number_per_layer
        mask[layer_id][neuron_id] = 1.
        
    return mask

def load_importance(pkl_file_path):
    with open(pkl_file_path, "rb") as file:
        importances = pickle.load(file)
    return importances

def get_score(retain_pkl, forget_pkl):
    ri = load_importance(retain_pkl)
    fi = load_importance(forget_pkl)
    score = [x / (y + 0.01) for x, y in zip(fi, ri)]
    return score


class CustomOpt(AdamW):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, mask=None):
        super(CustomOpt, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.mask = mask
        self.neuron_name = "mlp.down_proj"

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        mask = self.mask.to(torch.cuda.current_device())

        for idx, group in enumerate(self.param_groups):
            #print(group['name'])
            for p in group['params']:
                #grad = p.grad
                #grad = p.grad*-torch.ones_like(mask[idx])
                grad = p.grad*mask[idx]
                p.data.add_(-group['lr'], grad)

        return loss

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        use_fast=False,
    )

    # data
    def convert_to_features(example_batch):
        input_encodings = tokenizer.batch_encode_plus(example_batch['text'], pad_to_max_length=True, max_length=512)
        target_encodings = tokenizer.batch_encode_plus(example_batch['text'], pad_to_max_length=True, max_length=512)

        encodings = {
            'input_ids': input_encodings['input_ids'],
            'labels': input_encodings['input_ids'].copy()
        }
        
        return encodings

    def combine_text(example):
        example["text"] = example["prompt"]["text"] + example["continuation"]["text"]
        return example

    
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_dataset("allenai/real-toxicity-prompts", split='train').shuffle(seed=42).select(range(10000))

    if data_args.retain_only:
        #print("train with only non toxic continuation")
        #train_dataset = train_dataset.filter(lambda example: example["continuation"]["toxicity"] is None or example["continuation"]["toxicity"] <= 0.5)
        print("train with only toxic continuation")
        train_dataset = train_dataset.filter(lambda example: example["continuation"]["toxicity"] is not None and example["continuation"]["toxicity"] > 0.5)

    train_dataset = train_dataset.map(combine_text, load_from_cache_file=False)
    train_dataset = train_dataset.map(convert_to_features, batched=True, load_from_cache_file=False)   
    test_dataset = train_dataset

    print(f"train_dataset size: {len(train_dataset)}")

    # model
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    score = get_score(data_args.retain_importances_pkl, data_args.forget_importances_pkl)
    mask = get_mask(model, score, pruning_percent=training_args.pruning_percent)

    #mask = torch.zeros(32, 5636096)

    trainer = CustomTrainer(model=model, tokenizer=tokenizer, args=training_args, mask=mask, train_dataset=train_dataset, eval_dataset=test_dataset)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    
if __name__ == "__main__":
    train()
