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
    ):
        self.model = model
        self.opt = opt
        self.device = device
        self.alpha = None
        self.xmin = None
        self.param_num = len(parameters)
        # print("param_num: ", self.param_num)
        self.param_list = parameters

    def calc_importance(self, dataloader: DataLoader, imp_type: str) -> List:
        importance = [0] * self.param_num

        dataloader = tqdm(dataloader)
        dataloader.set_description("Calculating importance...")
        
        if imp_type == "perturb":
            with torch.no_grad():
                for batch in dataloader:
                    b = {k: v.to(self.device) for k, v in batch.items()}     # a dictionary of tensors
                    origin_loss = self.model(**b).loss.item()

                    for (idx, item) in enumerate(self.param_list):
                        weight_name = item + ".weight"
                        bias_name = item + ".bias"
                        origin_state = self.model.state_dict()
                        new_state = self.model.state_dict()
                        if weight_name in origin_state:
                            new_state[weight_name].data = -new_state[weight_name].data
                        if bias_name in origin_state:
                            new_state[bias_name].data = -new_state[bias_name].data
                        self.model.load_state_dict(new_state)
                        importance[idx] += self.model(**b).loss.item() - origin_loss
                        
                        # recover the original parameter
                        self.model.load_state_dict(origin_state)

        # elif imp_type == 'grad':
        #     for batch in dataloader:
        #         b = {k: v.to(self.device) for k, v in batch.items()}
        #         loss = self.model(**b).loss
        #         # loss = torch.tensor(self.model(**b).loss, requires_grad=True, device=self.device)
        #         loss.requires_grad = True
        #         self.opt.zero_grad()
        #         loss.backward()

        #         idx = 0
        #         for (name, p) in self.model.named_parameters():
        #             if p.grad is not None:
        #                 importance[idx] += p.grad.data.clone().pow(2)
        #                 idx += 1

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
            id = score_pair[i][1]
            name = self.param_list[id]
            print(name)
            with torch.no_grad():
                weight_name = name + ".weight"
                bias_name = name + ".bias"
                new_state = self.model.state_dict()
                if weight_name in self.model.state_dict():
                    new_state[weight_name].zero_()
                if bias_name in self.model.state_dict():
                    new_state[bias_name].zero_()
                self.model.load_state_dict(new_state)
                
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