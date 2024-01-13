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

    def calc_importance(self, dataloader: DataLoader) -> List:

        importance = [0] * len(self.model.named_parameters())

        dataloader = tqdm(dataloader)
        dataloader.set_description("Calculating importance...")
        
        with torch.no_grad():
            for batch in dataloader:
                b = {k: v.to(self.device) for k, v in batch.items()}     # a dictionary of tensors

                idx = 0
                for (name, p) in self.model.named_parameters():
                    p.data = -p.data
                    importance[idx] += self.model(**b).loss.item()
                    idx += 1
                    p.data = -p.data
                    
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
        pruning_number = int(self.total_neuron_number * pruning_percent)

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
