"""
The importance in this file is calculated per parameter, which means larger granularity.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, dataset
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import time
import copy
import os
import pdb
import math
import shutil
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
# import seaborn as sns
import scipy.stats as stats
from typing import Dict, List

import transformers
# from transformers import GPT2LMHeadModel

import gc

# neuron_name: str = "mlp.c_proj",
class ParameterPerturber:
    def __init__(
        self,
        model,
        opt,
        device="cuda" if torch.cuda.is_available() else "cpu",
        parameters=None,
        neuron_name: str = "mlp.c_proj",
    ):
        self.model = model
        self.opt = opt
        self.device = device
        self.alpha = None
        self.xmin = None
        self.param_num = sum(1 for _ in model.named_parameters())
        # print("param_num: ", self.param_num)
        self.name_list = []

    def calc_importance(self, dataloader: DataLoader, imp_type: str) -> List:
        importance = [0] * self.param_num

        dataloader = tqdm(dataloader)
        dataloader.set_description("Calculating importance...")
        
        if imp_type == "perturb":
            with torch.no_grad():
                for batch in dataloader:
                    b = {k: v.to(self.device) for k, v in batch.items()}     # a dictionary of tensors
                    origin_loss = self.model(**b).loss.item()

                    idx = 0
                    for (name, p) in self.model.named_parameters():
                        p.data = -p.data
                        importance[idx] += self.model(**b).loss.item() - origin_loss
                        idx += 1
                        p.data = -p.data
        elif imp_type == 'grad':
            for batch in dataloader:
                b = {k: v.to(self.device) for k, v in batch.items()}
                loss = self.model(**b).loss
                self.opt.zero_grad()
                loss.backward()

                idx = 0
                for (name, p) in self.model.named_parameters():
                    if p.grad is None:
                        importance[idx] += p.grad.data.clone().pow(2)
                        idx += 1

        return importance

    def modify_neuron(
        self,
        score: list,
        pruning_percent: float,
    ) -> None:
        """
        change neuron weights by score

        Returns:
        None

        """
        pruning_number = int(self.param_num * pruning_percent)

        score_pair = [(s, id) for id, s in enumerate(score)]
        score_pair.sort(key=lambda x: x[0], reverse=True)   # true means descending

        for i in tqdm(range(pruning_number)):
            del_pair = score_pair[i]
            id = del_pair[1]
            with torch.no_grad():
                idx = 0
                for (name, p) in self.model.named_parameters():
                    if idx == id:
                        p.zero_()
                    idx += 1
                
                
        return None

    def freeze_params(
            self,
            score: list,
            pruning_percent: float,
        ) -> None:
            
            pruning_number = int(self.param_num * pruning_percent)

            score_pair = [(s, id) for id, s in enumerate(score)]
            score_pair.sort(key=lambda x: x[0], reverse=True)   # true means descending

            for i in tqdm(range(pruning_number)):
                del_pair = score_pair[i]
                id = del_pair[1]
                with torch.no_grad():
                    idx = 0
                    for (name, p) in self.model.named_parameters():
                        if idx == id:
                            self.name_list.append(name)
                        idx += 1
                    
            assert len(self.name_list) == pruning_number

            for (name, p) in self.model.named_parameters():
                if name not in self.name_list:
                    p.requires_grad = False

            return None