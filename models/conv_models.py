from torch import nn
from torchsummary import summary


class SimpleConv(nn.Module):
    def __init__(self, size_h, size_w, kernel_size=(3, 3), dropout=0.5, num_classes=2):
        super().__init__()
        linear_layer_size = 64 * (size_w - (kernel_size[0] - 1) * 4) * (size_h - (kernel_size[1] - 1) * 4)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(linear_layer_size, num_classes)
        )

    def forward(self, input):
        return self.model(input)


class BNConv(nn.Module):
    def __init__(self, size_h, size_w, kernel_size=(3, 3), dropout=0.5, num_classes=2):
        super().__init__()
        linear_layer_size = 64 * (size_w - (kernel_size[0] - 1) * 4) * (size_h - (kernel_size[1] - 1) * 4)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=kernel_size),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=kernel_size),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel_size),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(p=dropout),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Flatten(),
            nn.Linear(linear_layer_size, num_classes)
        )

    def forward(self, input):
        return self.model(input)


if __name__ == '__main__':
    simple_conv_model = SimpleConv(size_h=96, size_w=96)
    print(summary(simple_conv_model, (3, 96, 96)))

    bn_conv_model = SimpleConv(size_h=96, size_w=96)
    print(summary(bn_conv_model, (3, 96, 96)))
