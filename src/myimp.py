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
                module.register_forward_hook(hook=hook)

        # children = self.model.children()
        # print(children)
        # for child in children:
        #     if isinstance(child, transformers.pytorch_utils.Conv1D):
        #         print("child is conv1d")
        #         child.register_forward_hook(hook=hook)

    def calc_importance(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
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
        for batch in dataloader:
            b = {k: v.to(self.device) for k, v in batch.items()}     # a dictionary of tensors
            self.model(**b)

            # 6 neurons
            # get length of feature_out
            # if it's a tuple, try to get the first element by [0]
            # print(len(self.feature_in))
            # print(type(self.feature_in[0]))
            # print(type(self.feature_in[0][0]))
            # print(self.feature_in[0][0].shape)
            # print(len(self.feature_out))
            # print(type(self.feature_out[0]))
            # print(self.feature_out[0].shape)
            # breakpoint()

        # in fact we should consider the batchsize = 1
        # D_num = the number of samples
        D_num = len(dataloader)
        # print(D_num)
        importance = []
        # print(len(self.feature_out))
        # breakpoint()

        def check_zero(x):
            x = x[0]
            total_cnt = []
            row = x.shape[0]    # different channels?
            col = x.shape[1]    # the number of neurons
            for j in range(col):
                cnt = 0
                for i in range(row):
                    if x[i, j] > 0:
                        cnt += 1
                total_cnt.append(cnt)    
            return total_cnt, row

        row = self.feature_out[0][0].shape[0]
        col = self.feature_out[0][0].shape[1]
        print(row, col)
        # 6 c_proj layers
        for i in range(6):
            total_cnt_list = [0] * col
            total_sum = 0
            for j in range(D_num):
                # print(type(self.feature_out[6 * j + i]))
                # print(self.feature_out[6 * j + i].shape)
                numlist, tnum = check_zero(self.feature_out[6 * j + i])
                # print(len(numlist))
                total_cnt_list = [x + y for x, y in zip(total_cnt_list, numlist)]
                print("total, ", len(total_cnt_list))
                total_sum += tnum
            
            # print(total_cnt_list)
            # print(total_sum)
            for item in total_cnt_list:
                item /= total_sum
            importance.extend(total_cnt_list)
        print(len(importance))      # stands for the total neuron number
        # print(importance)
        # breakpoint()

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
            for j in range(3072):
                self.model.transformer.h[layer_id].mlp.c_proj.weight[j][neuron_id] = 0
            self.model.transformer.h[layer_id].mlp.c_proj.bias[neuron_id] = 0

        return None


###############################################
