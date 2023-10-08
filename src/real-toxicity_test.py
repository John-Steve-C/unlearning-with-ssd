"""
Refer to forget_full_class_... for comments
This file is near identical with minimal modifications to facilitate random forgetting.
Seperate file to allow for easy reuse.
"""

import random
import os
import wandb

# import optuna
from typing import Tuple, List
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, dataset
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import models
from unlearn import *
from utils import *
import forget_random_strategies
import datasets
import models
import conf
from training_utils import *

import transformers
from transformers import AutoTokenizer, get_scheduler, AutoModelForCausalLM
from datasets import load_dataset

# from optimum.gptq import GPTQQuantizer, load_quantized_model

"""
Get Args
"""
parser = argparse.ArgumentParser()
parser.add_argument("-origin_model", type=str, required=True, help="origin model without training")
parser.add_argument(
    "-model_name_or_path",
    type=str,
    required=True,
    help="Path to model",
)
parser.add_argument(
    "-dataset",
    type=str,
    required=True,
    nargs="?",
    choices=["skg/toxigen-data", "allenai/real-toxicity-prompts"],
    help="dataset to train on",
)
parser.add_argument("-classes", type=int, required=True, help="number of classes")
# parser.add_argument("-gpu", action="store_true", default=False, help="use gpu or not")
parser.add_argument("-b", type=int, default=128, help="batch size for dataloader")
parser.add_argument("-warm", type=int, default=1, help="warm up training phase")
parser.add_argument("-lr", type=float, default=0.1, help="initial learning rate")
parser.add_argument(
    "-method",
    type=str,
    required=True,
    nargs="?",
    choices=[
        "baseline",
        "retrain",
        "finetune",
        "blindspot",
        "amnesiac",
        "FisherForgetting",
        "pdr_tuning",
    ],
    help="select unlearning method from choice set",
)
parser.add_argument(
    "-forget_perc", type=float, required=True, help="Percentage of trainset to forget"
)
parser.add_argument(
    "-epochs", type=int, default=1, help="number of epochs of unlearning method to use"
)
parser.add_argument("-seed", type=int, default=0, help="seed for runs")
args = parser.parse_args()

# ---------------------------------------- Set seeds

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
transformers.set_seed(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------- get tokenizer & quantized model

tokenizer = transformers.AutoTokenizer.from_pretrained(
    args.model_name_or_path,
    padding_side="right",
    use_fast=False,
)
tokenizer.pad_token = tokenizer.eos_token

# dataset = ["auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]
# gptq_config = GPTQConfig(bits=4, dataset = dataset, tokenizer=tokenizer)

model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
model.to(device)

unlearning_teacher = AutoModelForCausalLM.from_pretrained(args.origin_model)
unlearning_teacher.to(device)

#------------------------- data preprocess

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


# trainset = getattr(datasets, args.dataset)(
#     root=root, download=True, train=True, unlearning=True, img_size=img_size
# )
# validset = getattr(datasets, args.dataset)(
#     root=root, download=True, train=False, unlearning=True, img_size=img_size
# )

trainset = load_dataset(args.dataset, split='train')
validset = load_dataset(args.dataset, split='train')
trainset = trainset.map(combine_text, load_from_cache_file=False)
trainset = trainset.map(convert_to_features, batched=True)
validset = validset.map(combine_text, load_from_cache_file=False)
validset = validset.map(convert_to_features, batched=True)

trainset = trainset.remove_columns(["prompt", "text", "filename", "begin", "end", "challenging"])
validset = validset.remove_columns(["prompt", "text", "filename", "begin", "end", "challenging"])
trainset.set_format('torch')
validset.set_format('torch')

# no random split
forget_train = trainset.filter(lambda example: example["continuation"]["toxicity"] is not None and example["continuation"]["toxicity"] > 0.5)
retain_train = trainset.filter(lambda example: example["continuation"]["toxicity"] is None or example["continuation"]["toxicity"] <= 0.5)

trainset = trainset.remove_columns(["continuation"])
validset = validset.remove_columns(["continuation"])
forget_train = forget_train.remove_columns(["continuation"])
retain_train = retain_train.remove_columns(["continuation"])

trainloader = DataLoader(trainset, num_workers=4, batch_size=args.b, shuffle=True)
validloader = DataLoader(validset, num_workers=4, batch_size=args.b, shuffle=False)
# forget_train, retain_train = torch.utils.data.random_split(
#     trainset, [args.forget_perc, 1 - args.forget_perc]
# )
forget_train_dl = DataLoader(list(forget_train), batch_size=128)
retain_train_dl = DataLoader(list(retain_train), batch_size=128, shuffle=True)
forget_valid_dl = forget_train_dl
retain_valid_dl = validloader

full_train_dl = DataLoader(
    ConcatDataset((retain_train_dl.dataset, forget_train_dl.dataset)),
    batch_size=args.b,
)

# --------------------------------------- parameters

model_size_scaler = 1

kwargs = {
    "model": model,
    "unlearning_teacher": unlearning_teacher,
    "retain_train_dl": retain_train_dl,
    "retain_valid_dl": retain_valid_dl,
    "forget_train_dl": forget_train_dl,
    "forget_valid_dl": forget_valid_dl,
    "full_train_dl": full_train_dl,
    "valid_dl": validloader,
    "dampening_constant": 1,
    "selection_weighting": 10 * model_size_scaler,
    "num_classes": args.classes,
    "dataset_name": args.dataset,
    "device": device,
    "model_name": args.origin_model,
}

wandb.init(
    project=f"{args.origin_model}_real-toxicity_test",
    name=f"{args.method}",
)

# -------------------------------------------------------- executing the method
import time
from tqdm import tqdm

start = time.time()
testacc, retainacc, zrf, mia, d_f = getattr(forget_random_strategies, args.method)(     # execution
    **kwargs
)
end = time.time()
time_elapsed = end - start

print(testacc, retainacc, zrf, mia)
wandb.log(
    {
        "TestAcc": testacc,
        "RetainTestAcc": retainacc,
        "ZRF": zrf,
        "MIA": mia,
        "Df": d_f,
        "model_scaler": model_size_scaler,
        "MethodTime": time_elapsed,  # do not forget to deduct baseline time from it to remove results calc (acc, MIA, ...)
    }
)

wandb.finish()