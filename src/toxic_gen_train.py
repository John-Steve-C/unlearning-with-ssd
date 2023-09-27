import datasets
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import argparse
import math
import transformers
from transformers import AutoTokenizer

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="distilgpt2")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


import torch
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(device)


from transformers import AutoModelForCausalLM


from transformers import Trainer, TrainingArguments

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
        input_encodings = tokenizer.batch_encode_plus(example_batch['text'], pad_to_max_length=True, max_length=12)
        target_encodings = tokenizer.batch_encode_plus(example_batch['text'], pad_to_max_length=True, max_length=12)

        encodings = {
            'input_ids': input_encodings['input_ids'],
            'labels': input_encodings['input_ids'].copy()
        }
        
        return encodings

    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = load_dataset(data_args.data_path, split='train').select(range(100))
    print(train_dataset.column_names)
    train_dataset = train_dataset.map(convert_to_features, batched=True, load_from_cache_file=False)
    print(train_dataset.column_names)

    test_dataset = load_dataset(data_args.data_path, split='test').select(range(100))
    test_dataset = test_dataset.map(convert_to_features, batched=True, load_from_cache_file=False)

    # model
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    
if __name__ == "__main__":
    train()
