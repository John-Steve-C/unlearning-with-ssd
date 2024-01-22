# From https://github.com/vikram2000b/bad-teaching-unlearning

import torch
from torch import nn
from torch.nn import functional as F
from training_utils import *
from transformers import AutoTokenizer, RobertaTokenizer, RobertaForSequenceClassification
from tqdm import tqdm
import pickle


TOKENIZER_PATH = os.environ.get("LLAMA_TOKENIZER_PATH", "distilgpt2")
print(f"###### Using tokenizer: {TOKENIZER_PATH}")
orig_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, padding_side="right", use_fast=False)

CLASSIFIER_PATH = os.environ.get("CLASSIFIER_PATH", "SkolkovoInstitute/roberta_toxicity_classifier")
print(f"###### Using classifier: {CLASSIFIER_PATH}")
toxic_cls_tokenizer = RobertaTokenizer.from_pretrained(CLASSIFIER_PATH)
toxic_cls_model = RobertaForSequenceClassification.from_pretrained(CLASSIFIER_PATH)

# if CLASSIFIER_PATH == "SkolkovoInstitute/roberta_toxicity_classifier":
#     toxic_cls_tokenizer.save_pretrained('./models/roberta_toxicity_classifier')
#     toxic_cls_model.save_pretrained('./models/roberta_toxicity_classifier')

toxic_cls_model.to('cuda')
toxic_cls_model.eval()

@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    # outputs = [validation_step(model, batch, device) for batch in val_loader]
    
    # print("start evaluating...")
    val_loader = tqdm(val_loader)
    val_loader.set_description("Evaluating...")
    outputs = [my_valid(model, batch, device, i) for i, batch in enumerate(val_loader)]
    # print("finish evaluating...")

    return validation_epoch_end(model, outputs)

def judge_toxicity(text, device):
    # load tokenizer and model weights
    # prepare the input
    batch = toxic_cls_tokenizer.encode(text, return_tensors='pt', max_length=512)
    # inference
    batch = batch.to(device)
    output = toxic_cls_model(batch)
    prob = torch.softmax(output.logits, dim=1)[:, 1]
    # batch = batch.to('cpu')
    del batch
    return torch.argmax(output.logits, dim=-1).detach().cpu(), prob.detach().cpu()

def get_toxicity_score(preds, labels, device):
    # slice the text to 512 tokens so that it can be fed into the judge_toxicity function
    pred_text = [orig_tokenizer.decode(pred).replace('<|endoftext|>', '')[:512] for pred in preds]
    label_text = [orig_tokenizer.decode(label).replace('<|endoftext|>', '')[:512] for label in labels]
    # print(pred_text[0])
    # print(label_text[0])
    # breakpoint()
    cnt = 0
    total_pred_prob = 0
    for pred, label in zip(pred_text, label_text):
        pred_toxicity, pred_prob = judge_toxicity(pred, device)
        label_toxicity, label_prob = judge_toxicity(label, device)
        # print("pred_toxicity ", pred_toxicity)
        # print("label_toxicity ", label_toxicity)
        if pred_toxicity == label_toxicity:
            cnt += 1
        total_pred_prob += pred_prob
    return torch.tensor(cnt / len(preds)).detach().cpu() * 100, total_pred_prob / len(preds)

def my_valid(model, batch, device, idx):
    # torch.cuda.empty_cache()
    # print("now valid : ", idx)
    input_ids, labels = batch.values()
    # labels = labels.to(device)
    b = {k: v.to(device) for k, v in batch.items()}     # a dictionary of tensors
    out = model(**b)
    loss = out.loss     # Negative Log Likelihood Loss

    _, preds = torch.max(out.logits, dim=-1)            # max returns (value ,index)!
    # preds = preds.to(device)
    toxic_acc, toxic_prob = get_toxicity_score(preds, labels, device)
    # perplexity = torch.exp(loss)

    # labels = labels.to('cpu')
    # preds = preds.to('cpu')
    # del labels
    # del preds
    torch.cuda.empty_cache()

    return {"Loss": loss.detach().cpu(), "Acc": toxic_acc, "Toxic_Level": toxic_prob}

# def calculate_perplexity():

#-------------------------------------------------------------------------------------------------------

# it's not correct due to we can't simply compare the logits & labels for text generation
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=-1)
    # dim = [1, 512] as embeddings
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

    # print("input_ids", input_ids.shape)
    # print("labels", labels.shape)
    # print("out.logits", out.logits.shape)

    acc = accuracy(out.logits, labels)  # Calculate accuracy

    return {"Loss": loss.detach(), "Acc": acc}


def validation_epoch_end(model, outputs):
    batch_losses = [x["Loss"] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
    batch_accs = [x["Acc"] for x in outputs]
    batch_toxicities = [x["Toxic_Level"] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
    epoch_toxicity = torch.stack(batch_toxicities).mean()
    return {"Loss": epoch_loss.item(), "Acc": epoch_acc.item(), "Toxic_Level": epoch_toxicity.item()}


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


# @torch.no_grad()
# def evaluate(model, val_loader, device):
#     model.eval()
#     outputs = [validation_step(model, batch, device) for batch in val_loader]
#     return validation_epoch_end(model, outputs)


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

def reverse_fit(
    epochs, model, train_loader, val_loader, device, lr=0.01, milestones=None
):
    torch.cuda.empty_cache()
    history = []

    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)
    # if milestones:
    #     train_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #         optimizer, milestones=milestones, gamma=0.2
    #     )  # learning rate decay
    #     warmup_scheduler = WarmUpLR(optimizer, len(train_loader))

    for epoch in range(epochs):
        # if epoch > 1 and milestones:
        #     train_scheduler.step(epoch)

        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = training_step(model, batch, device)
            train_losses.append(loss)
            loss.backward()

            for (name, p) in model.named_parameters():
                if p.grad is not None and p.requires_grad:
                    p.grad.data = -p.grad.data
            
            optimizer.step()
            optimizer.zero_grad()

            lrs.append(get_lr(optimizer))

            # if epoch <= 1 and milestones:
            #     warmup_scheduler.step()

        # Validation phase
        result = evaluate(model, val_loader, device)
        result["train_loss"] = torch.stack(train_losses).mean().item()
        result["lrs"] = lrs
        epoch_end(model, epoch, result)
        # history.append(result)
    # return history

def reverse_part_fit(
    epochs, mask, model, train_loader, val_loader, device, lr=0.01, milestones=None
):
    torch.cuda.empty_cache()
    history = []

    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = training_step(model, batch, device)
            train_losses.append(loss)
            loss.backward()

            idx = 0
            for (name, p) in model.named_parameters():
                if p.grad is not None and p.requires_grad:
                    # print(idx, name)
                    # print(p.grad.data.shape)
                    # (weight, bias) = (0, 1), (2, 3) ... they are arranged in order
                    p.grad.data *= -mask[int(idx / 2)]  # reverse the grad of some neurons in the parameter
                    idx += 1
            
            optimizer.step()
            optimizer.zero_grad()

            lrs.append(get_lr(optimizer))

        # Validation phase
        result = evaluate(model, val_loader, device)
        result["train_loss"] = torch.stack(train_losses).mean().item()
        result["lrs"] = lrs
        epoch_end(model, epoch, result)
        # history.append(result)
    # return history

def get_importance(pkl_name, pdr, dataloader, forget_type, load_from_file):

    if load_from_file and os.path.exists(pkl_name):
        with open(pkl_name, "rb") as file:
            imp = pickle.load(file)
    else:
        with torch.no_grad():
            imp = pdr.calc_importance(dataloader, forget_type)
        with open(pkl_name, "wb") as file:
            pickle.dump(imp, file)
    
    return imp