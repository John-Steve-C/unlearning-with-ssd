import torch
import datasets
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import argparse
import math
import transformers
from transformers import AutoTokenizer, get_scheduler
# import ssd
from tqdm.auto import tqdm

# from metrics import *
# from unlearn import *
# from utils import *


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
    training_args = TrainingArguments(
        bf16=False,                      # enable bf16 training
        output_dir='./results',          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=8,   # batch size per device during training
        per_device_eval_batch_size=8,    # batch size for evaluation
        gradient_accumulation_steps=1,   # number of updates steps to accumulate before performing a backward/update pass
        evaluation_strategy="steps",     # evaluation strategy to adopt during training
        eval_steps=100,                  # number of update steps between two evaluations
        save_strategy="steps",           # checkpoint save strategy
        save_steps=2000,                 # number of updates steps before two checkpoint saves
        save_total_limit=1,              # limit the total amount of checkpoints. Deletes the older checkpoints in increments of save_steps
        learning_rate=2e-5,              # learning rate
        weight_decay=0.,                 # strength of weight decay
        warmup_ratio=0.03,               # number of warmup steps for learning rate scheduler
        lr_scheduler_type='cosine',      # learning rate scheduler
        logging_steps=10,                # logging steps
        tf32=False,                      # enable tf32 precision
    )
    model_args = ModelArguments(model_name_or_path="distilgpt2")
    data_args = DataArguments(data_path="skg/toxigen-data")
    
    # parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        use_fast=False,
    )

    # data
    def convert_to_features(example_batch):
        input_encodings = tokenizer.batch_encode_plus(example_batch['text'], pad_to_max_length=True, max_length=12)
        target_encodings = tokenizer.batch_encode_plus(example_batch['text'], pad_to_max_length=True, max_length=12)
        target_group_encodings = tokenizer.batch_encode_plus(example_batch['target_group'], pad_to_max_length=True, max_length=12)
        # factual_group_encodings = tokenizer.batch_encode_plus(example_batch['factual?'], pad_to_max_length=True, max_length=12)
        # ingroup_effect_encodings = tokenizer.batch_encode_plus(example_batch['ingroup_effect'], pad_to_max_length=True, max_length=12)
        
        # lewd_encodings = tokenizer.batch_encode_plus(example_batch['lewd'], pad_to_max_length=True, max_length=12),
        # framing_encodings = tokenizer.batch_encode_plus(example_batch['framing'], pad_to_max_length=True, max_length=12),
        # predicted_group_encodings = tokenizer.batch_encode_plus(example_batch['predicted_group'], pad_to_max_length=True, max_length=12),
        # stereotyping_encodings = tokenizer.batch_encode_plus(example_batch['stereotyping'], pad_to_max_length=True, max_length=12),
        # predicted_author_encodings = tokenizer.batch_encode_plus(example_batch['predicted_author'], pad_to_max_length=True, max_length=12),
        # actual_method_encodings = tokenizer.batch_encode_plus(example_batch['actual_method'], pad_to_max_length=True, max_length=12),

        encodings = {
            'input_ids': input_encodings['input_ids'],
            'labels': input_encodings['input_ids'].copy(),
            # 'target_group_ids': target_group_encodings['input_ids'],
            # 'factual_group_ids': factual_group_encodings['input_ids'],
            # 'ingroup_effect_ids': ingroup_effect_encodings['input_ids'],
            
            # 'lewd_ids': lewd_encodings['input_ids'],
            # 'framing_ids': framing_encodings['input_ids'],
            # 'predicted_group_ids': predicted_group_encodings['input_ids'],
            # 'stereotyping_ids': stereotyping_encodings['input_ids'],
            # 'predicted_author_ids': predicted_author_encodings['input_ids'],
            # 'actual_method_ids': actual_method_encodings['input_ids'],
        }
        
        return encodings
    
    # another convert to features function
    # def tokenize_function(examples):
    #     return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = load_dataset(data_args.data_path, split='train')#.select(range(100))
    print(train_dataset.column_names)
    train_dataset = train_dataset.map(convert_to_features, batched=True, load_from_cache_file=False)
    
    test_dataset = load_dataset(data_args.data_path, split='test')#.select(range(100))
    print(train_dataset.column_names)
    test_dataset = test_dataset.map(convert_to_features, batched=True, load_from_cache_file=False)

    train_dataset = train_dataset.remove_columns(["text", "target_group", "factual?", "ingroup_effect", "lewd", "framing", "predicted_group", "stereotyping", "predicted_author", "actual_method",  "intent", "toxicity_ai", "toxicity_human"])
    test_dataset = test_dataset.remove_columns(["text", "target_group", "factual?", "ingroup_effect", "lewd", "framing", "predicted_group", "stereotyping", "predicted_author", "actual_method",  "intent", "toxicity_ai", "toxicity_human"])
    # train_dataset = train_dataset.rename_column("labels", "labels")
    # test_dataset = test_dataset.rename_column("labels", "labels")
    print(train_dataset.column_names)
    train_dataset.set_format('torch')
    test_dataset.set_format('torch')

    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size=training_args.per_device_eval_batch_size, shuffle=True)

    # model
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    model.to(device)
    num_training_steps = len(train_data) * training_args.num_train_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
    
    lr_scheduler = get_scheduler(
        name='linear',optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(training_args.num_train_epochs):
        for batch in train_data:
            b = {k: v.to(device) for k, v in batch.items()}     # a dictionary of tensors
            outputs = model(**b)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    # tokenizer.save_pretrained("./trained_model")
    model.save_pretrained("./trained_model")

if __name__ == "__main__":
    train()
                                                        