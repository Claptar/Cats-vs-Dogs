from torch import nn, optim


class SimpleFC(nn.Module):
    def __init__(self, in_features, dropout=0.5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1)
        )

    def forward(self, input):
        return self.model(input)


class SimpleFCBN(nn.Module):
    def __init__(self, in_features, dropout=0.5, num_classes=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, input):
        return self.model(input)
