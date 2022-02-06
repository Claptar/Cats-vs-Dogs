import torch
from torch import optim, nn
from models import SimpleFC, SimpleFCBN, SimpleConv
from dataset import CatDogDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from train import train_loop
import wandb
from torchsummary import summary
import os

APIKEY = os.environ.get('WANDBAPIKEY')

EPOCH_NUM = 2
BATCH_SIZE = 256
SIZE_H = 96
SIZE_W = 96

image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transformer = transforms.Compose([
    transforms.Resize((SIZE_H, SIZE_W)),        # scaling images to fixed size
    transforms.ToTensor(),                      # converting to tensors
    transforms.Normalize(image_mean, image_std) # normalize image data per-channel
])


if __name__ == '__main__':
    # Loading and processing data
    train_dataset = CatDogDataset(folder_path='/home/user/PycharmProjects/Cats-vs-Dogs/data/train/',
                                  transforms=transformer)
    test_dataset = CatDogDataset(folder_path='/home/user/PycharmProjects/Cats-vs-Dogs/data/test/',
                                 transforms=transformer)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

    # Initializing model
    lr = 1e-3
    in_features = 3 * SIZE_W * SIZE_H
    dropout = 0.5
    model = SimpleConv(size_h=SIZE_H, size_w=SIZE_W, dropout=dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print('Model architecture:')
    print(summary(model, (3, SIZE_H, SIZE_W), device=device))

    # configurate wandb
    config = {
             "Size_H/Size_W": SIZE_H,
             "dropuot": dropout,
             "lr": lr,
             "Batch_size": BATCH_SIZE
             }

    # training
    wandb.login(key=APIKEY)
    wandb.init(project="my-test-project", entity="claptar", name="Pycharm", config=config)
    train_loop(model, criterion, optimizer, train_loader, test_loader, n_epoch=EPOCH_NUM)
    wandb.finish()

