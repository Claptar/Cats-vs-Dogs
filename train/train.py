import torch
from torch import optim
from tqdm import tqdm
import wandb
import numpy as np


def train_epoch(model, criterion, optimizer, dataloader, device):
    loss_log = []
    for batch_idx, (data, target) in tqdm(enumerate(dataloader)):
        # making predictions and calculating loss
        predictions = model(data.to(device))
        loss = criterion(predictions, target.to(device))
        # logging loss
        loss_log.append(loss.item())
        wandb.log({'batch_loss': loss.item()})
        # making backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss_log


def train_loop(model, criterion, optimizer, train_loader, test_loader, device, n_epoch=2):
    for epoch in range(n_epoch):
        loss_log = train_epoch(model, criterion, optimizer, train_loader)
        train_accuracy = compute_accuracy(model, train_loader)
        test_accuracy = compute_accuracy(model, test_loader)
        wandb.log({
            'mean_epoch_loss': np.array(loss_log).mean(),
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy
        })


def compute_accuracy(model, dataloader):
    return 'Not implemented yet'


