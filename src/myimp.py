"""
This file is used for the Selective Synaptic Dampening method
Strategy files use the methods from here
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

###############################################
# Clean implementation
###############################################

def clean_list(lists):
    # print(type(lists))
    for item in lists:
        # print(type(item)) # tuple
        for data in item:
            data.detach()
        del item

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

        # print(parameters)
        self.lower_bound = parameters["lower_bound"]
        self.exponent = parameters["exponent"]
        self.magnitude_diff = parameters["magnitude_diff"]  # unused
        self.min_layer = parameters["min_layer"]
        self.max_layer = parameters["max_layer"]
        self.forget_threshold = parameters["forget_threshold"]
        self.dampening_constant = parameters["dampening_constant"]
        self.selection_weighting = parameters["selection_weighting"]

        # hook feature outputs
        # store the outputs will cost CUDA out of memory?
        # self.module_name = []
        self.feature_in = []
        self.feature_out = []
        self.hooks = []

        def hook(module, fea_in, fea_out):
            # print("hooker working")
            # self.module_name.append(module.__class__)
            self.feature_in.append(fea_in)
            self.feature_out.append(fea_out)
            # print(fea_in[0].shape)
            # print(fea_out.shape)
            return None

        neuron = "mlp.c_proj"
        # neuron = "mlp.dropout"
        # neuron = "attn.c_proj"
        # neuron = "ln_f"
        for (name, module) in self.model.named_modules():
            if neuron in name: #and module.__class__ == transformers.pytorch_utils.Conv1D:
                # print(name)
                self.hooks.append(module.register_forward_hook(hook=hook))

        # children = self.model.children()
        # print(children)
        # for child in children:
        #     if isinstance(child, transformers.pytorch_utils.Conv1D):
        #         print("child is conv1d")
        #         child.register_forward_hook(hook=hook)

    def calc_importance(self, dataloader: DataLoader, imp_type: str) -> Dict[str, torch.Tensor]:
        """
        Adapated from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Calculate per-parameter, importance
            returns a dictionary [param_name: list(importance per parameter)]
        Parameters:
        DataLoader (DataLoader): DataLoader to be iterated over
        Returns:
        importance (dict(str, torch.Tensor([]))): named_parameters-like dictionary containing list of importances for each parameter
        """
        # criterion = nn.CrossEntropyLoss()
        self.feature_in = []
        self.feature_out = []
        total_cnt_list = [0] * (768 * 6)
        # batch_size = dataloader.batch_size
        D_num = len(dataloader) * dataloader.batch_size     # the number of samples

        dataloader = tqdm(dataloader)
        dataloader.set_description("Calculating importance...")
        for batch in dataloader:
            b = {k: v.to(self.device) for k, v in batch.items()}     # a dictionary of tensors
            self.model(**b)

            # get length of feature_out
            # if it's a tuple, try to get the first element by [0]
            # print(len(self.feature_in))
            # print(type(self.feature_in[0]))
            # print(type(self.feature_in[0][0]))
            # print(self.feature_in[0][0].shape)
            # print(len(self.feature_out))
            # print(type(self.feature_out))   # list, length = 6, stands for 6 mlp blocks (layers)
            # print(type(self.feature_out[0]))
            # print(self.feature_out[0].shape)
            # print(len(dataloader))
            # breakpoint()

            row = self.feature_out[0][0].shape[0]   # channels=512
            col = self.feature_out[0][0].shape[1]   # neurons=768
            # print(self.feature_out[0].device)
            # print(row, col)
            f = torch.stack(self.feature_out, dim=0)    # 6 * batch_size * 512 * 768
            # print(f.shape)
            f = f.permute(0, 3, 1, 2)   # 6 * 768 * batch_size * 512
            for i in range(6):                      # layers
                for j in range(768):            # neurons
                    if imp_type == 'freq':
                        total_cnt_list[i * 768 + j] += torch.sum(torch.gt(f[i][j], 0))
                    elif imp_type == 'abs':
                        total_cnt_list[i * 768 + j] += torch.sum(torch.abs(f[i][j]))
                    elif imp_type == 'rms':
                        total_cnt_list[i * 768 + j] += torch.sum(torch.square(f[i][j]))
                    else:   # imp_type == "std"
                        total_cnt_list[i * 768 + j] += torch.sum(torch.square(f[i][j] - torch.mean(f[i][j])))

            # for num in range(batch_size):    # each data
            #     for i in range(6):                      # layers
            #         for j in range(768):                # neurons
            #             total_cnt = 0
            #             for k in range(512):            # channels
            #                 if self.feature_out[i][num][k][j] > 0:
            #                     total_cnt += 1
            #             total_cnt_list[i * 768 + j] += total_cnt  

            # del mask
            del f
            clean_list(self.feature_in)
            clean_list(self.feature_out)
            torch.cuda.empty_cache()

            self.feature_in = []
            self.feature_out = []

        # print(len(self.feature_out))
        # breakpoint()

        if imp_type == "freq" or imp_type == "abs":
            importance = [x / (row * D_num) for x in total_cnt_list]
        else:
            importance = [torch.sqrt(x / (row * D_num)) for x in total_cnt_list]
        # print(len(importance))      # stands for the total neuron number = 768 * 6 = 4608
        return importance

    def modify_neuron(
        self,
        score: list,
        neuron_number_per_layer: int = 768,
        pruning_number: int = 50,
    ) -> None:
        """
        change neuron weights by score

        Returns:
        None

        """
        score_pair = [(s, id) for id, s in enumerate(score)]
        score_pair.sort(key=lambda x: x[0], reverse=True)   # true means descending

        for i in range(pruning_number):
            del_pair = score_pair[i]
            id = del_pair[1]
            neuron_id = id % neuron_number_per_layer
            layer_id = id // neuron_number_per_layer
            with torch.no_grad():
                for j in range(3072):
                    self.model.transformer.h[layer_id].mlp.c_proj.weight[j][neuron_id].zero_()
                self.model.transformer.h[layer_id].mlp.c_proj.bias[neuron_id].zero_()
                # we need to prevent the gradient of the pruned neuron from being updated
                # self.model.transformer.h[layer_id].mlp.c_proj.weight.requires_grad = False
                # self.model.transformer.h[layer_id].mlp.c_proj.bias.requires_grad = False

        # remember to remove hook
        for hook in self.hooks:
            hook.remove()

        return None

   
    # TODO: still need to modify
    # we can't directly set a neuron's 'requires_grad' to False because it's not a leaf variable
    def freeze_neuron(
        self,
        score: list,
        neuron_number_per_layer: int = 768,
        pruning_number: int = 50,
    ) -> None:
        score_pair = [(s, id) for id, s in enumerate(score)]
        score_pair.sort(key=lambda x: x[0], reverse=True)   # true means descending

        # freeze less important neurons in forget set
        for i in range(pruning_number, len(score_pair)):
            del_pair = score_pair[i]
            id = del_pair[1]
            neuron_id = id % neuron_number_per_layer
            layer_id = id // neuron_number_per_layer
            with torch.no_grad():
                for j in range(3072):
                    self.model.transformer.h[layer_id].mlp.c_proj.weight[j][neuron_id].zero_()
                # we need to prevent the gradient of the pruned neuron from being updated
                self.model.transformer.h[layer_id].mlp.c_proj.weight.requires_grad = False
                self.model.transformer.h[layer_id].mlp.c_proj.bias[neuron_id].requires_grad = False

        # remember to remove hook
        for hook in self.hooks:
            hook.remove()

        return None