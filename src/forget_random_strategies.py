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
import copy

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
import myimp_new as imp_new
import myimp_perturb as imp_perturb
import myimp_mix as imp_mix
import conf
import math

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
    torch.cuda.empty_cache()
    
    # batch size need to transfer from main
    # zrf = UnLearningScore(model, unlearning_teacher, forget_valid_dl, 1, device)
    zrf = 0
    # torch.cuda.empty_cache()

    # mia = get_membership_attack_prob(retain_train_dl, forget_train_dl, valid_dl, model)
    mia = 0
    # torch.cuda.empty_cache()

    loss_acc_dict = evaluate(model, valid_dl, device)
    torch.cuda.empty_cache()
    retain_acc_dict = evaluate(model, retain_valid_dl, device)
    torch.cuda.empty_cache()    
    forget_acc_dict = evaluate(model, forget_valid_dl, device)
    torch.cuda.empty_cache()
    

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
    _ = fit_one_cycle(
        1, model, retain_train_dl, retain_valid_dl, lr=1e-4, device=device
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
    model.eval()
    
    sample_importances = pdr.calc_importance(forget_train_dl)
    print(sample_importances)
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
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

    print(kwargs)
    neuron_name = kwargs["neuron_name"]
    modify_method = kwargs["modify_method"]

    pdr = imp.ParameterPerturber(model, optimizer, device)
    pdr.freeze_neurons()
    model.eval()
    
    retain_importances = get_importance(kwargs["retain_importances_pkl"], pdr, retain_train_dl, kwargs["forget_type"], kwargs["load_from_file"])
    forget_importances = get_importance(kwargs["forget_importances_pkl"], pdr, forget_train_dl, kwargs["forget_type"], kwargs["load_from_file"])
    
    pdr.remove_hooks()

    score = [x / (y + 0.01) for x, y in zip(forget_importances, retain_importances)]    

    if modify_method == 'zero': 
        # modify method 1
        pdr.modify_neuron(score, pruning_percent=kwargs["pruning_percent"])
    elif modify_method == 'reverse':
        # modify method 2 : reverse grad
        mask = pdr.get_mask(score, pruning_percent=kwargs["pruning_percent"])
        reverse_part_fit(2, mask, model, forget_train_dl, forget_valid_dl, device)

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

    # print(kwargs)
    # neuron_name = kwargs["neuron_name"]
    modify_method = kwargs["modify_method"]

    pdr = imp_large.ParameterPerturber(model, optimizer, device)
    model.eval()
        
    retain_importances = get_importance(kwargs["retain_importances_pkl"], pdr, retain_train_dl, kwargs["forget_type"], kwargs["load_from_file"])
    print("re, ", retain_importances)
    forget_importances = get_importance(kwargs["forget_importances_pkl"], pdr, forget_train_dl, kwargs["forget_type"], kwargs["load_from_file"])
    print("ff, ", forget_importances)

    score = [x / (y + 0.01) for x, y in zip(forget_importances, retain_importances)]
    print(score)

    if modify_method == 'zero': 
        # modify method 1
        pdr.modify_neuron(score, pruning_percent=kwargs["pruning_percent"])
    elif modify_method == 'reverse':
        # modify method 2 : reverse grad
        pdr.freeze_params(score, pruning_percent=kwargs["pruning_percent"])
        reverse_fit(5, model, forget_train_dl, forget_valid_dl, device)


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

## mixture pruning 1, perturb(large) + abs(small)...

# def mixture_pruning(
#     model,
#     unlearning_teacher,
#     retain_train_dl,
#     retain_valid_dl,
#     forget_train_dl,
#     forget_valid_dl,
#     valid_dl,
#     dampening_constant,
#     selection_weighting,
#     full_train_dl,
#     device,
#     **kwargs,
# ):
#     # load the trained model
#     optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

#     print(kwargs)
#     modify_method = kwargs["modify_method"]

#     pdr_1 = imp_large.ParameterPerturber(model, optimizer, device)
#     model.eval()
        
#     retain_importances = get_importance(kwargs["retain_importances_pkl"], pdr_1, retain_train_dl, 'perturb', kwargs["load_from_file"])
#     forget_importances = get_importance(kwargs["forget_importances_pkl"], pdr_1, forget_train_dl, 'perturb', kwargs["load_from_file"])

#     score_1 = [x / (y + 0.01) for x, y in zip(forget_importances, retain_importances)]
#     param_list = pdr_1.get_important_param(score_1, pruning_percent=0.1)
    
#     orig_param_list = list(param_list)            # deep copy
#     print('origin: ', orig_param_list)
#     for i in range(len(param_list)):
#         param_list[i], _, _ = param_list[i].rpartition('.')   # remove the last '.'
#     param_list = list(dict.fromkeys(param_list))  # remove duplicate
#     print('now: ', param_list)

#     # param_list = ['transformer.wte', 'transformer.h.0.attn.c_attn', 'transformer.h.0.attn.c_proj']

#     pdr_2 = imp.ParameterPerturber(model, optimizer, device, param_list)
#     # pdr.freeze_neurons()
    
#     retain_importances = get_importance('mixture_imp_retain', pdr_2, retain_train_dl, kwargs["forget_type"], load_from_file=False)
#     forget_importances = get_importance('mixture_imp_forget', pdr_2, forget_train_dl, kwargs["forget_type"], load_from_file=False)
    
#     pdr_2.remove_hooks()

#     score = [x / (y + 0.01) for x, y in zip(forget_importances, retain_importances)]    

#     if modify_method == 'zero': 
#         # modify method 1
#         pdr_2.modify_neuron(score, pruning_percent=kwargs["pruning_percent"], param_list=orig_param_list)
#     elif modify_method == 'reverse':
#         # modify method 2 : reverse grad
#         mask = pdr_2.get_mask(score, pruning_percent=kwargs["pruning_percent"])
#         reverse_part_fit(5, mask, model, forget_train_dl, forget_valid_dl, device)

#     return get_metric_scores(
#         model,
#         unlearning_teacher,
#         retain_train_dl,
#         retain_valid_dl,
#         forget_train_dl,
#         forget_valid_dl,
#         valid_dl,
#         device,
#     )

## mixture pruning 2, abs + perturb, both large granularity...

# def mixture_pruning(
#     model,
#     unlearning_teacher,
#     retain_train_dl,
#     retain_valid_dl,
#     forget_train_dl,
#     forget_valid_dl,
#     valid_dl,
#     dampening_constant,
#     selection_weighting,
#     full_train_dl,
#     device,
#     **kwargs,
# ):
#     # load the trained model
#     optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

#     print(kwargs)
#     modify_method = kwargs["modify_method"]

#     pdr_1 = imp_new.ParameterPerturber(model, optimizer, device)
#     model.eval()
        
#     retain_importances = get_importance(kwargs["retain_importances_pkl"], pdr_1, retain_train_dl, kwargs["forget_type"], kwargs["load_from_file"])
#     forget_importances = get_importance(kwargs["forget_importances_pkl"], pdr_1, forget_train_dl, kwargs["forget_type"], kwargs["load_from_file"])

#     score_1 = [x / (y + 0.01) for x, y in zip(forget_importances, retain_importances)]
#     param_list = pdr_1.get_important_param(score_1, pruning_percent=kwargs["pruning_percent"])
    
#     # score_pair = [(s, id) for id, s in enumerate(score_1)]
#     # print('before ', score_pair)
#     # score_pair.sort(key=lambda x: x[0], reverse=True)   # true means descending
#     # print('after ', score_pair)

#     pdr_1.remove_hooks()

#     # orig_param_list = list(param_list)            # deep copy
#     print('first filter important parameters: ', param_list)

#     pdr_2 = imp_perturb.ParameterPerturber(model, optimizer, device, param_list)
#     # pdr.freeze_neurons()
    
#     retain_importances = get_importance('mixture_imp_retain', pdr_2, retain_train_dl, 'perturb', kwargs["load_from_file"])
#     forget_importances = get_importance('mixture_imp_forget', pdr_2, forget_train_dl, 'perturb', kwargs["load_from_file"])
    
#     # print('retain_importances: ', retain_importances)
#     # print('forget_importances: ', forget_importances)

#     score = [x / (y + 0.01) for x, y in zip(forget_importances, retain_importances)]    

#     # score_pair = [(s, id) for id, s in enumerate(score)]
#     # print('before ', score_pair)
#     # score_pair.sort(key=lambda x: x[0], reverse=True)   # true means descending
#     # print('after ', score_pair)

#     print('modify at last: ')
#     pdr_2.modify_neuron(score, pruning_percent=kwargs["pruning_percent_2"])

#     return get_metric_scores(
#         model,
#         unlearning_teacher,
#         retain_train_dl,
#         retain_valid_dl,
#         forget_train_dl,
#         forget_valid_dl,
#         valid_dl,
#         device,
#     )

## mixture pruning 3, combined importance

def mixture_pruning(
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
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    print(kwargs)
    modify_method = kwargs["modify_method"]

    pdr = imp_mix.ParameterPerturber(model, optimizer, device)
    # because the model now is wrapped by DeepspeedEngine, and the .eval() member is None
    model.eval()

    # if kwargs["load_from_file"]:
    #     with open(kwargs["forget_importances_pkl"], "rb") as file:
    #         forget_imp1 = pickle.load(file)
    #     with open(kwargs["forget_importances_pkl_2"], "rb") as file:
    #         forget_imp2 = pickle.load(file)
        
    #     print("load importance from file, length = ", len(forget_imp1), len(forget_imp2))
    # else:
    #     forget_imp1, forget_imp2 = pdr.calc_importance(forget_train_dl, kwargs["forget_type"])

    #     with open(kwargs["forget_importances_pkl"], "wb") as file:
    #         pickle.dump(forget_imp1, file)
    #     with open(kwargs["forget_importances_pkl_2"], "wb") as file:
    #         pickle.dump(forget_imp2, file)
    #     print("calculate importance, length = ", len(forget_imp1), len(forget_imp2)) 

    forget_imp1 = pdr.calc_importance(forget_train_dl, kwargs["forget_type"])
    with open(kwargs["forget_importances_pkl"], "wb") as file:
        pickle.dump(forget_imp1, file)
    
    with open(kwargs["forget_importances_pkl_2"], "rb") as file:
        forget_imp2 = pickle.load(file)
    forget_importances = [0] * len(forget_imp2)

    # allocate abs importance to parameters
    # make len(forget_imp_1) = len(forget_imp_2)
    idx = 0
    for i in range(len(forget_imp1)):
        module_name = pdr.module_list[i][0]
        while idx < len(forget_imp2) and module_name not in pdr.actual_param_list[idx]:
            idx += 1
        while idx < len(forget_imp2) and module_name in pdr.actual_param_list[idx]:
            forget_importances[idx] += forget_imp1[i]
            idx += 1

    # print('forget_importances: ', forget_importances)
    # print('forget_imp2: ', forget_imp2)

    forget_importances = normalize_list(forget_importances)
    forget_imp2 = normalize_list(forget_imp2)

    # print('forget_importances: ', forget_importances)
    # print('forget_imp2: ', forget_imp2)

    score = [(x + y) for x, y in zip(forget_importances, forget_imp2)]

    pdr.remove_hooks()
    pdr.modify_neuron(score, pruning_percent=kwargs["pruning_percent"])

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
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    # pdr = imp.ParameterPerturber(model, optimizer, device)

    # pdr.freeze_neurons()

    # ------------------------------------------------------------------------------

    for (name, p) in model.named_parameters():
        if 'mlp.c_proj' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
    reverse_fit(5, model, forget_train_dl, forget_valid_dl, device)

    # -----------------------------------------------------------------------------

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

# a more effect way to reverse grad
def vector_negation(
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

    new_param = model.state_dict()
    orig_param = copy.deepcopy(new_param)

    # print("*1", orig_param)
    # print("test on finetune model 1: ", evaluate(model, valid_dl, device))


    _ = fit_one_cycle(
        1, model, forget_train_dl, forget_valid_dl, lr=1e-4, device=device
    )

    print("test on finetune model: ", evaluate(model, valid_dl, device))
    # print("*2", orig_param)
    # print("*3", new_param)


    for name in orig_param.keys():
        orig_param[name] -= new_param[name] - orig_param[name]

    # print("*4", orig_param)

    model.load_state_dict(orig_param)

    torch.cuda.empty_cache()

    # pdr.remove_hooks()

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
