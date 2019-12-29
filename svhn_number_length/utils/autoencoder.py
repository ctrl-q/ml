from torch import nn


class AutoEncoder(nn.Module):
    """Convolutional AutoEncoder used for transform"""

    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, 6),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, return_indices=True)
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 3, stride=1),
            nn.ReLU(True)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(16, 32, 5, stride=2),
            nn.ReLU(True)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 6, stride=2, padding=2),
            nn.ReLU(True)
        )
        self.decoder4 = nn.Tanh()

    def forward(self, x):
        x, indices = self.encoder(x)
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.decoder4(x)
        return x
