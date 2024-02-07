"""
The importance in this file is calculated per neuron, which means smaller granularity.
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

###############################################
# Clean implementation
###############################################

def has_children(module):
    return any(module.children())

# neuron_name: str = "mlp.c_proj",
class ParameterPerturber:
    def __init__(
        self,
        model,
        opt,
        device="cuda" if torch.cuda.is_available() else "cpu",
        param_name_list=['transformer.h.0.mlp.c_proj', 'transformer.h.1.mlp.c_proj', 'transformer.h.2.mlp.c_proj', 'transformer.h.3.mlp.c_proj', 'transformer.h.4.mlp.c_proj', 'transformer.h.5.mlp.c_proj'],
    ):
        self.model = model
        self.opt = opt
        self.device = device

        # hook feature outputs
        # store the outputs will cost CUDA out of memory?
        # self.module_name = []
        self.feature_in = []
        self.feature_out = []
        self.hooks = []
        self.module_list = []

        def hook(module, fea_in, fea_out):
            # print("hooker working")
            # self.module_name.append(module.__class__)
            # self.feature_in.append(fea_in)
            self.feature_out.append(fea_out)
            # print(fea_in[0].shape)
            # print(fea_out.shape)
            return None

        self.param_name_list = param_name_list
        self.total_param_number = 0
        self.neuron_number_per_layer = []

        for item in self.model.named_modules():
            name = item[0]
            module = item[1]
            if not has_children(module): #and module.__class__ == transformers.pytorch_utils.Conv1D:
                # print(module.weight.shape)
                # print(name)
                # print(module)
                self.total_param_number += 1
                self.module_list.append(item)
                self.hooks.append(module.register_forward_hook(hook=hook))

        print('total :', self.total_param_number)
        # print("layers: ", layers)
        # print("weight shape: ", self.weight_shape)

        # children = self.model.children()
        # print(children)
        # for child in children:
        #     if isinstance(child, transformers.pytorch_utils.Conv1D):
        #         print("child is conv1d")
        #         child.register_forward_hook(hook=hook)

    def calc_importance(self, dataloader: DataLoader, imp_type: str) -> List:
        self.feature_in = []
        self.feature_out = []
        total_cnt_list = [0] * self.total_param_number
        D_num = len(dataloader) # * dataloader.batch_size     # the number of samples

        dataloader = tqdm(dataloader)
        dataloader.set_description("Calculating importance...")
        for batch in dataloader:
            b = {k: v.to(self.device) for k, v in batch.items()}     # a dictionary of tensors
            self.model(**b)

            f = self.feature_out
            for i in range(self.total_param_number):
                if imp_type == 'freq':
                    total_cnt_list[i] += torch.sum(torch.gt(f[i], 0)) / f[i].numel()
                elif imp_type == 'abs':
                    total_cnt_list[i] += torch.sum(torch.abs(f[i])) / f[i].numel()
                elif imp_type == 'rms':
                    total_cnt_list[i] += torch.sum(torch.square(f[i])) / f[i].numel()
                else:   # imp_type == "std"
                    total_cnt_list[i] += torch.sum(torch.square(f[i] - torch.mean(f[i]))) / f[i].numel()

            # del mask
            del f
            gc.collect()
            torch.cuda.empty_cache()

            self.feature_in = []
            self.feature_out = []

        if imp_type == "freq" or imp_type == "abs":
            importance = [x / D_num for x in total_cnt_list]
        else:
            importance = [math.sqrt(x / D_num) for x in total_cnt_list]
        # print("total neuron number: ", len(importance))      # stands for the total neuron number
        return importance
    
    def modify_neuron(
        self,
        score: list,
        pruning_percent: float,
        param_list = None,
    ) -> None:
        """
        change neuron weights by score

        Returns:
        None

        """
        pruning_number = int(self.total_param_number * pruning_percent)

        score_pair = [(s, id) for id, s in enumerate(score)]
        score_pair.sort(key=lambda x: x[0], reverse=True)   # true means descending

        for i in tqdm(range(pruning_number)):
            del_pair = score_pair[i]
            id = del_pair[1]
            layer_id, neuron_id = self.get_pos(id)
            with torch.no_grad():
                # for j in range(self.weight_shape[0]):
                #     self.model.transformer.h[layer_id].mlp.c_proj.weight[j][neuron_id].zero_()
                # self.model.transformer.h[layer_id].mlp.c_proj.bias[neuron_id].zero_()
                
                if param_list is not None:
                    name = self.module_list[layer_id][0]
                    if name + '.weight' in param_list:
                        for j in range(self.channels):
                            self.module_list[layer_id][1].weight[j][neuron_id].zero_()
                    if name + '.bias' in param_list:
                        self.module_list[layer_id][1].bias[neuron_id].zero_()
                else:
                    for j in range(self.channels):
                        self.module_list[layer_id][1].weight[j][neuron_id].zero_()
                    try:
                        self.module_list[layer_id][1].bias[neuron_id].zero_()
                    except:
                        continue
                    
                # we need to prevent the gradient of the pruned neuron from being updated
                # self.model.transformer.h[layer_id].mlp.c_proj.weight.requires_grad = False
                # self.model.transformer.h[layer_id].mlp.c_proj.bias.requires_grad = False

        return None

    def get_pos(
        self,
        id
    ):
        # neuron_id = id % self.neuron_number_per_layer
        # layer_id = id // self.neuron_number_per_layer
        idx = 0
        while id >= self.neuron_number_per_layer[idx]:
            id -= self.neuron_number_per_layer[idx]
            idx += 1
        layer_id = idx
        neuron_id = id
        return layer_id, neuron_id

   
    # we can't directly set a neuron's 'requires_grad' to False because it's not a leaf variable
    def freeze_neurons(
        self,
    ) -> None:
        """
        freeze neurons except the self.neuron_name
        """
        
        # name_list = []
        # for module in self.module_list:
        #     for p in module.named_parameters():
        #         name_list.append(p[0])

        for (name, p) in self.model.named_parameters():
            if self.neuron_name not in name:
                # if len(p) > 2:
                #     print(len(p))
                #     for i in range(len(p)):
                #         print(p[i])

                p.requires_grad = False

        return None

    def remove_hooks(
        self,
    ):
        for hook in self.hooks:
            hook.remove()

        return None

    def get_important_param(
        self, 
        score: list,
        pruning_percent: float,
    ) -> List:
        pruning_number = int(self.total_param_number * pruning_percent)

        score_pair = [(s, id) for id, s in enumerate(score)]
        score_pair.sort(key=lambda x: x[0], reverse=True)   # true means descending

        param_list = []

        for i in tqdm(range(pruning_number)):
            id = score_pair[i][1]
            param_list.append(self.module_list[id][0])  # get name

        return param_list
