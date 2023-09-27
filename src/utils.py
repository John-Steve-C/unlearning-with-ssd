# From https://github.com/vikram2000b/bad-teaching-unlearning

import torch
from torch import nn
from torch.nn import functional as F
from training_utils import *
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilgpt2", padding_side="right", use_fast=False)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=-1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds)) * 100


def training_step(model, batch, device):
    b = {k: v.to(device) for k, v in batch.items()}     # a dictionary of tensors
    out = model(**b)
    loss = out.loss
    return loss


def validation_step(model, batch, device):
    input_ids, labels = batch.values()
    labels = labels.to(device)
    # images, clabels = batch
    # images, clabels = images.to(device), clabels.to(device)
    # out = model(images)  # Generate predictions
    # loss = F.cross_entropy(out, clabels)  # Calculate loss
    b = {k: v.to(device) for k, v in batch.items()}     # a dictionary of tensors
    out = model(**b)
    loss = out.loss
    labels = labels.to(device)
    #import pdb; pdb.set_trace()
    # print("input_ids", input_ids.shape)
    # print("labels", labels.shape)
    # print("out.logits", out.logits.shape)

    acc = accuracy(out.logits, labels)  # Calculate accuracy

    return {"Loss": loss.detach(), "Acc": acc}


def validation_epoch_end(model, outputs):
    batch_losses = [x["Loss"] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
    batch_accs = [x["Acc"] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
    return {"Loss": epoch_loss.item(), "Acc": epoch_acc.item()}


def epoch_end(model, epoch, result):
    print(
        "Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch,
            result["lrs"][-1],
            result["train_loss"],
            result["Loss"],
            result["Acc"],
        )
    )


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    outputs = [validation_step(model, batch, device) for batch in val_loader]
    return validation_epoch_end(model, outputs)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def fit_one_cycle(
    epochs, model, train_loader, val_loader, device, lr=0.01, milestones=None
):
    torch.cuda.empty_cache()
    history = []

    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)
    if milestones:
        train_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.2
        )  # learning rate decay
        warmup_scheduler = WarmUpLR(optimizer, len(train_loader))

    for epoch in range(epochs):
        if epoch > 1 and milestones:
            train_scheduler.step(epoch)

        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = training_step(model, batch, device)
            train_losses.append(loss)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            lrs.append(get_lr(optimizer))

            if epoch <= 1 and milestones:
                warmup_scheduler.step()

        # Validation phase
        result = evaluate(model, val_loader, device)
        result["train_loss"] = torch.stack(train_losses).mean().item()
        result["lrs"] = lrs
        epoch_end(model, epoch, result)
        history.append(result)
    return history
