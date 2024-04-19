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

        self.actual_param_list = []
        for name, _ in model.named_parameters():
            # filter embedding layer
            if 'wte' in name or 'wpe' in name or 'embed_tokens' in name:
                continue
            self.actual_param_list.append(name)

        self.total_param_number = len(self.actual_param_list)
        self.total_module_number = 0        # actually the number of sub-modules
        self.neuron_number_per_layer = []

        for item in self.model.named_modules():
            name = item[0]
            module = item[1]

            # filter embedding layer
            if 'wte' in name or 'wpe' in name or 'embed_tokens' in name:
                continue

            weight_name = name + '.weight'
            bias_name = name + '.bias'
            if not has_children(module) and (weight_name in self.actual_param_list or bias_name in self.actual_param_list): #and module.__class__ == transformers.pytorch_utils.Conv1D:
                # print(module.weight.shape)
                # print(name)
                # print(module)
                self.total_module_number += 1
                self.module_list.append(item)
                self.hooks.append(module.register_forward_hook(hook=hook))

        print('total module number :', self.total_module_number)
        print('total param number :', self.total_param_number)

        # children = self.model.children()
        # print(children)
        # for child in children:
        #     if isinstance(child, transformers.pytorch_utils.Conv1D):
        #         print("child is conv1d")
        #         child.register_forward_hook(hook=hook)

    def calc_importance(self, dataloader: DataLoader, imp_type: str) -> List:
        self.feature_in = []
        self.feature_out = []
        total_cnt_list = [0] * self.total_module_number
        imp_2 = [0] * self.total_param_number
        D_num = len(dataloader) # * dataloader.batch_size     # the number of samples

        dataloader = tqdm(dataloader)
        dataloader.set_description("Calculating importance...")
        for batch in dataloader:
            b = {k: v.to(self.device) for k, v in batch.items()}     # a dictionary of tensors
            loss = self.model(**b).loss

            with torch.no_grad():
                f = self.feature_out
                for i in range(self.total_module_number):
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

            # now we calculate the grad outside the program
            # # calculate grad importance
            # self.opt.zero_grad()

            # # loss.requires_grad = True
            # loss.backward()
            # # self.model.backward(loss)

            # with torch.no_grad():
            #     idx = 0
            #     for (name, p) in self.model.named_parameters():
            #         if p.grad is not None and name in self.actual_param_list:
            #             imp_2[idx] += torch.norm(p.grad.data.clone(), 2)
            #             # imp_2[idx] += p.grad.data.clone().pow(2)
            #             idx += 1

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
        
        print("pruning parameters: ")
        
        pruning_number = int(self.total_param_number * pruning_percent)

        score_pair = [(s, id) for id, s in enumerate(score)]
        score_pair.sort(key=lambda x: x[0], reverse=True)   # true means descending

        with torch.no_grad():
            new_state = self.model.state_dict()
            for i in tqdm(range(pruning_number)):
                id = score_pair[i][1]
                name = self.actual_param_list[id]
                print(name)
                new_state[name].zero_()
            self.model.load_state_dict(new_state)
                
        return None

    def remove_hooks(
        self,
    ):
        for hook in self.hooks:
            hook.remove()

        return None
