import torch
from torch import optim
from tqdm import tqdm


def train_epoch(model, criterion, optimizer, dataloader):
    loss_log = []
    for batch_idx, (data, target) in tqdm(dataloader):
        # making predictions and calculating loss
        predictions = model(data)
        loss = criterion(predictions, target)
        # making backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


