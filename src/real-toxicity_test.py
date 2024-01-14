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
from datasets import load_dataset, concatenate_datasets

# from optimum.gptq import GPTQQuantizer, load_quantized_model

"""
Get Args
"""
parser = argparse.ArgumentParser()
parser.add_argument("-origin_model", type=str, default="distilgpt2", required=False, help="origin model without training")
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
parser.add_argument("-classes", type=int, default=2, required=False, help="number of classes")
parser.add_argument("-use_sample", action="store_true", default=False, help="use a subset of training set for dev")
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
        "imp_pruning",
        "reverse_gradient",
        "imp_pruning_large",
    ],
    help="select unlearning method from choice set",
)
parser.add_argument(
    "-forget_perc", type=float, default=0.0, required=False, help="Percentage of trainset to forget"
)
parser.add_argument(
    "-epochs", type=int, default=1, help="number of epochs of unlearning method to use"
)
parser.add_argument("-seed", type=int, default=0, help="seed for runs")
parser.add_argument("-pruning_percent", type=float, default=0.5, help="percentage of weights to prune")
parser.add_argument(
    "-forget_type", type=str, default="freq", help="forget type: freq/abs/rms/std"
)

parser.add_argument(
    "-forget_importances_pkl", type=str, default="None", help="list of importance"
)

parser.add_argument(
    "-retain_importances_pkl", type=str, default="None", help="list of importance"
)

parser.add_argument(
    "-neuron_name", type=str, default="mlp.c_proj", help="mlp.down_proj for llama2"
)

parser.add_argument(
    "-modify_method", type=str, default="zero", help="reverse for reverse gradient"
)

args = parser.parse_args()

# ---------------------------------------- Set seeds

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
transformers.set_seed(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# --------------------------------------- get tokenizer & quantized model

tokenizer = transformers.AutoTokenizer.from_pretrained(
    args.model_name_or_path,
    padding_side="right",
    use_fast=False,
)
tokenizer.pad_token = tokenizer.eos_token

# quantizer = GPTQQuantizer(bits=4, dataset="c4") # block_name_to_quantize = "model.decoder.layers", model_seqlen = 2048
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
#print(model)
# print(model.config)
# print(model.transformer.h[0].attn.c_attn.weight.shape)
# print(model.transformer.h[0].attn.c_proj.weight.shape)
# print(model.transformer.h[0].mlp.c_fc.weight.shape)
# print(model.transformer.h[0].mlp.c_proj.weight.shape)
# model = quantizer.quantize_model(model, tokenizer)
model.to(device)

# or we can call it origin model
#unlearning_teacher = AutoModelForCausalLM.from_pretrained(args.origin_model)
# unlearning_teacher = quantizer.quantize_model(unlearning_teacher, tokenizer)
#unlearning_teacher.to(device)

# disable unlearning_teacher to same gpu memory
unlearning_teacher = None

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
    example["text"] = example["prompt"]["text"] # + example["continuation"]["text"]
    return example


#trainset = load_dataset(args.dataset, split='train').shuffle(seed=42).select(range(2 * total_size))
trainset = load_dataset(args.dataset, split='train')
# validset = load_dataset(args.dataset, split='train').select(range(10000, 12000))
validset = trainset
trainset = trainset.map(combine_text)
trainset = trainset.map(convert_to_features, batched=True)
validset = validset.map(combine_text)
validset = validset.map(convert_to_features, batched=True)

trainset = trainset.remove_columns(["text", "filename", "begin", "end", "challenging"])
validset = validset.remove_columns(["text", "filename", "begin", "end", "challenging"])
trainset.set_format('torch')
validset.set_format('torch')

# forget toxic data
forget_train = trainset.filter(lambda example: example["continuation"]["toxicity"] is not None and example["continuation"]["toxicity"] > 0.5)
retain_train = trainset.filter(lambda example: example["continuation"]["toxicity"] is None or example["continuation"]["toxicity"] <= 0.5)

# select forget_perc of toxic data
#forget_train = forget_train.select(range(int(total_size * args.forget_perc)))
#retain_train = retain_train.select(range(int(total_size * (1 - args.forget_perc))))

if args.use_sample:
    forget_train = forget_train.select(range(int(256)))
    retain_train = retain_train.select(range(int(256)))

print('total dataset size : ', len(trainset))
print('forget train size : ', len(forget_train))
print('retain train size : ', len(retain_train))

# valid on non-toxic data
# forget_valid = validset.filter(lambda example: example["continuation"]["toxicity"] is not None and example["continuation"]["toxicity"] > 0.5)
# retain_valid = validset.filter(lambda example: example["continuation"]["toxicity"] is not None and example["continuation"]["toxicity"] <= 0.5)

trainset = trainset.remove_columns(["prompt", "continuation"])
validset = validset.remove_columns(["prompt", "continuation"])
forget_train = forget_train.remove_columns(["prompt", "continuation"])
retain_train = retain_train.remove_columns(["prompt", "continuation"])

trainloader = DataLoader(trainset, num_workers=4, batch_size=args.b, shuffle=True)
validloader = DataLoader(validset, num_workers=4, batch_size=args.b, shuffle=False)
# forget_train, retain_train = torch.utils.data.random_split(
#     trainset, [args.forget_perc, 1 - args.forget_perc]
# )
forget_train_dl = DataLoader(list(forget_train), batch_size=args.b, num_workers=8, pin_memory=True)
retain_train_dl = DataLoader(list(retain_train), batch_size=args.b, num_workers=8, pin_memory=True)
forget_valid_dl = forget_train_dl
retain_valid_dl = retain_train_dl

full_train_dl = DataLoader(
    ConcatDataset((retain_train_dl.dataset, forget_train_dl.dataset)),
    batch_size=args.b,
)
print('actual total train size : ', len(full_train_dl.dataset))

trainloader = full_train_dl
validloader = full_train_dl

# --------------------------------------- parameters

# alpha(selection_weighting) is affected by dataset size
model_size_scaler = 1 # alpha = 10 * scaler?

kwargs = {
    "model": model,
    "unlearning_teacher": unlearning_teacher,
    "retain_train_dl": retain_train_dl,
    "retain_valid_dl": retain_valid_dl,
    "forget_train_dl": forget_train_dl,
    "forget_valid_dl": forget_valid_dl,
    "full_train_dl": full_train_dl,
    "valid_dl": validloader,
    "dampening_constant": 1,            # lambda
    "selection_weighting": 1,           # alpha
    "num_classes": args.classes,
    "dataset_name": args.dataset,
    "device": device,
    "model_name": args.origin_model,
    "pruning_percent": args.pruning_percent,
    "forget_type": args.forget_type,
    "retain_importances_pkl": args.retain_importances_pkl,
    "forget_importances_pkl": args.forget_importances_pkl,
    "neuron_name": args.neuron_name,
    "modify_method": args.modify_method
}

pure_model_name = args.origin_model.split("/")[-1]

# wandb.init(
#     project=f"{pure_model_name}_real-toxicity_test",
#     name=f"{args.method}",
# )

# -------------------------------------------------------- executing the method
import time
from tqdm import tqdm

torch.cuda.empty_cache()

def print_kwargs(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")
print("===== args =========")
print_kwargs(**kwargs)

start = time.time()
totaltacc, retainacc, forgetacc, retainppl, forgetppl, toxic_level, zrf, mia = getattr(forget_random_strategies, args.method)(     # execution
    **kwargs
)
end = time.time()
time_elapsed = end - start

print(args.method, ": total_acc = ", totaltacc, ",retain_acc = ", retainacc, ",forget_acc = ", forgetacc, ",retain_ppl = ", retainppl, ",forget_ppl = ", forgetppl, ",toxic_level = ", toxic_level, ",zrf = ", zrf, ",mia = ", mia, ",time = ", time_elapsed)

print(f"{retainppl} {forgetppl} {toxic_level}")

# wandb.log(
#     {
#         "TestAcc": testacc,
#         "RetainTestAcc": retainacc,
#         "ZRF": zrf,
#         "MIA": mia,
#         "Df": d_f,
#         "model_scaler": model_size_scaler,
#         # "MethodTime": time_elapsed,  # do not forget to deduct baseline time from it to remove results calc (acc, MIA, ...)
#     }
# )

# wandb.finish()
