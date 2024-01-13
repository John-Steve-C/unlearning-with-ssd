"""
Refer to forget_full_class_... for comments
This file is near identical with minimal modifications to facilitate random forgetting.
Seperate file to allow for easy reuse.
"""
import os
import random
import numpy as np
from typing import Tuple, List
from copy import deepcopy

import torch
from torch.utils.data import DataLoader, ConcatDataset, dataset
from tqdm import tqdm

from sklearn import linear_model, model_selection

from unlearn import *
from metrics import UnLearningScore, get_membership_attack_prob
from utils import *
import ssd as ssd
import myimp as imp
import myimp_large as imp_large
import conf
import math
import pickle

def get_metric_scores(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
):
    loss_acc_dict = evaluate(model, valid_dl, device)
    torch.cuda.empty_cache()
    retain_acc_dict = evaluate(model, retain_valid_dl, device)
    torch.cuda.empty_cache()
    # batch size need to transfer from main
    # zrf = UnLearningScore(model, unlearning_teacher, forget_valid_dl, 1, device)
    zrf = 0
    # torch.cuda.empty_cache()
    forget_acc_dict = evaluate(model, forget_valid_dl, device)
    torch.cuda.empty_cache()
    # mia = get_membership_attack_prob(retain_train_dl, forget_train_dl, valid_dl, model)
    mia = 0

    return (loss_acc_dict["Acc"], retain_acc_dict["Acc"], forget_acc_dict["Acc"], math.exp(retain_acc_dict["Loss"]), math.exp(forget_acc_dict["Loss"]), loss_acc_dict["Toxic_Level"], zrf, mia)


def baseline(
    model,
    unlearning_teacher, # it stands for origin model without pretraining
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
    **kwargs,
):
    return get_metric_scores(
        unlearning_teacher,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


def retrain(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    dataset_name,
    model_name,
    device,
    **kwargs,
):
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    if model_name == "ViT":
        epochs = getattr(conf, f"{dataset_name}_{model_name}_EPOCHS")
        milestones = getattr(conf, f"{dataset_name}_{model_name}_MILESTONES")
    else:
        epochs = getattr(conf, f"{dataset_name}_EPOCHS")
        milestones = getattr(conf, f"{dataset_name}_MILESTONES")
    _ = fit_one_cycle(
        epochs,
        model,
        retain_train_dl,
        retain_valid_dl,
        milestones=milestones,
        device=device,
    )

    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )

def finetune(
    model,          # finetuned model from (real_toxicity_train.py)
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    full_train_dl,      # modified...
    valid_dl,
    device,
    **kwargs,
):
    # in fact we do finetune on the retain dataset. Maybe we should finetune on full dataset?
    # _ = fit_one_cycle(
    #     5, model, retain_train_dl, retain_valid_dl, lr=0.02, device=device
    # )

    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


def blindspot(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
    **kwargs,
):
    student_model = deepcopy(model)
    KL_temperature = 1
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.0001)
    retain_train_subset = random.sample(
        retain_train_dl.dataset, int(0.3 * len(retain_train_dl.dataset))
    )

    if kwargs["model_name"] == "ViT":
        b_s = 128  # lowered batch size from 256 (original) to fit into memory
    else:
        b_s = 256

    blindspot_unlearner(
        model=student_model,
        unlearning_teacher=unlearning_teacher,
        full_trained_teacher=model,
        retain_data=retain_train_subset,
        forget_data=forget_train_dl.dataset,
        epochs=1,
        optimizer=optimizer,
        lr=0.0001,
        batch_size=b_s,
        device=device,
        KL_temperature=KL_temperature,
    )

    return get_metric_scores(
        student_model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


def amnesiac(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    num_classes,
    device,
    **kwargs,
):
    unlearninglabels = list(range(num_classes))
    unlearning_trainset = []

    for x, _, clabel in forget_train_dl.dataset:
        rnd = random.choice(unlearninglabels)
        while rnd == clabel:
            rnd = random.choice(unlearninglabels)
        unlearning_trainset.append((x, _, rnd))

    for x, _, y in retain_train_dl.dataset:
        unlearning_trainset.append((x, _, y))

    unlearning_train_set_dl = DataLoader(
        unlearning_trainset, 128, pin_memory=True, shuffle=True
    )

    _ = fit_one_unlearning_cycle(
        3, model, unlearning_train_set_dl, retain_valid_dl, device=device, lr=0.0001
    )
    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


def FisherForgetting(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    num_classes,
    device,
    **kwargs,
):
    def hessian(dataset, model):
        model.eval()
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        loss_fn = nn.CrossEntropyLoss()

        for p in model.parameters():
            p.grad_acc = 0
            p.grad2_acc = 0

        for data, _, orig_target in tqdm(train_loader):
            data, orig_target = data.to(device), orig_target.to(device)
            output = model(data)
            prob = F.softmax(output, dim=-1).data

            for y in range(output.shape[1]):
                target = torch.empty_like(orig_target).fill_(y)
                loss = loss_fn(output, target)
                model.zero_grad()
                loss.backward(retain_graph=True)
                for p in model.parameters():
                    if p.requires_grad:
                        p.grad_acc += (orig_target == target).float() * p.grad.data
                        p.grad2_acc += prob[:, y] * p.grad.data.pow(2)

        for p in model.parameters():
            p.grad_acc /= len(train_loader)
            p.grad2_acc /= len(train_loader)

    def get_mean_var(p, is_base_dist=False, alpha=3e-6):
        var = deepcopy(1.0 / (p.grad2_acc + 1e-8))
        var = var.clamp(max=1e3)
        if p.size(0) == num_classes:
            var = var.clamp(max=1e2)
        var = alpha * var

        if p.ndim > 1:
            var = var.mean(dim=1, keepdim=True).expand_as(p).clone()
        if not is_base_dist:
            mu = deepcopy(p.data0.clone())
        else:
            mu = deepcopy(p.data0.clone())
        if p.ndim == 1:
            # BatchNorm
            var *= 10
        #         var*=1
        return mu, var

    for p in model.parameters():
        p.data0 = deepcopy(p.data.clone())

    hessian(retain_train_dl.dataset, model)

    fisher_dir = []
    alpha = 1e-6
    for i, p in enumerate(model.parameters()):
        mu, var = get_mean_var(p, False, alpha=alpha)
        p.data = mu + var.sqrt() * torch.empty_like(p.data0).normal_()
        fisher_dir.append(var.sqrt().view(-1).cpu().detach().numpy())
    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


def pdr_tuning(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    dampening_constant,
    selection_weighting,
    full_train_dl,
    device,
    **kwargs,
):
    parameters = {
        "lower_bound": 1,
        "exponent": 1,
        "magnitude_diff": None,
        "min_layer": -1,
        "max_layer": -1,
        "forget_threshold": 1,
        "dampening_constant": dampening_constant,
        "selection_weighting": selection_weighting,
    }

    # load the trained model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    pdr = ssd.ParameterPerturber(model, optimizer, device, parameters)
    model = model.eval()
    
    sample_importances = pdr.calc_importance(forget_train_dl)
    original_importances = pdr.calc_importance(full_train_dl)
    
    pdr.modify_weight(original_importances, sample_importances)
    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )

def imp_pruning(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    dampening_constant,
    selection_weighting,
    full_train_dl,
    device,
    **kwargs,
):
    # load the trained model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    print(kwargs)
    neuron_name = kwargs["neuron_name"]
    modify_method = kwargs["modify_method"]

    pdr = imp.ParameterPerturber(model, optimizer, device, neuron_name=neuron_name)
    pdr.freeze_neurons()
    model = model.eval()
    
    
    
    retain_importances_pkl = kwargs["retain_importances_pkl"]
    forget_importances_pkl = kwargs["forget_importances_pkl"]

    if os.path.exists(retain_importances_pkl):
        with open(retain_importances_pkl, "rb") as file:
           retain_importances = pickle.load(file)
    else:
        with torch.no_grad():
           retain_importances = pdr.calc_importance(retain_train_dl, kwargs["forget_type"])
        with open(retain_importances_pkl, "wb") as file:
           pickle.dump(retain_importances, file)
           
    if os.path.exists(forget_importances_pkl):
        with open(forget_importances_pkl, "rb") as file:
           forget_importances = pickle.load(file)
    else:                      
        with torch.no_grad():
           forget_importances = pdr.calc_importance(forget_train_dl, kwargs["forget_type"])

        with open(forget_importances_pkl, "wb") as file:
           pickle.dump(forget_importances, file)

    score = [x / (y + 0.01) for x, y in zip(forget_importances, retain_importances)]
    
    if modify_method == 'zero': 
        # modify method 1
        pdr.modify_neuron(score, pruning_percent=kwargs["pruning_percent"])
    elif modify_method == 'reverse':
        # modify method 2 : reverse grad
        mask = pdr.get_mask(score, pruning_percent=kwargs["pruning_percent"])
        reverse_part_fit(5, mask, model, forget_train_dl, forget_valid_dl, device)

    pdr.remove_hooks()

    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )

def imp_pruning_large(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    dampening_constant,
    selection_weighting,
    full_train_dl,
    device,
    **kwargs,
):
    # load the trained model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    print(kwargs)
    # neuron_name = kwargs["neuron_name"]
    # modify_method = kwargs["modify_method"]

    pdr = imp_large.ParameterPerturber(model, optimizer, device)
    model = model.eval()
        
    with torch.no_grad():
        retain_importances = pdr.calc_importance(retain_train_dl, kwargs["forget_type"])
        forget_importances = pdr.calc_importance(forget_train_dl, kwargs["forget_type"])

    score = [x / (y + 0.01) for x, y in zip(forget_importances, retain_importances)]
    
    pdr.modify_neuron(score, pruning_percent=kwargs["pruning_percent"])

    # if modify_method == 'zero': 
    #     # modify method 1
    #     pdr.modify_neuron(score, pruning_percent=kwargs["pruning_percent"])
    # elif modify_method == 'reverse':
    #     # modify method 2 : reverse grad
    #     mask = pdr.get_mask(score, pruning_percent=kwargs["pruning_percent"])
    #     reverse_part_fit(5, mask, model, forget_train_dl, forget_valid_dl, device)


    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


# a naive approach to reverse all grad
def reverse_gradient(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    dampening_constant,
    selection_weighting,
    full_train_dl,
    device,
    **kwargs,
):

    # load the trained model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    pdr = imp.ParameterPerturber(model, optimizer, device)

    pdr.freeze_neurons()
    reverse_fit(5, model, forget_train_dl, forget_valid_dl, device)

    torch.cuda.empty_cache()

    pdr.remove_hooks()

    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )
