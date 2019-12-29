import torch.nn as nn
import torch.nn.functional as F
from models.chenxi import svhn


class Simple_Model(nn.Module):

    def __init__(self):
        super(Simple_Model, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 6)
        self.fc1 = nn.Linear(16 * 10 * 10, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 7)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # 3 x 54 x 54 -> 6 x 50 x 50
        x = self.pool(x)            # 6 x 50 x 50 -> 6 x 25 x 25
        x = F.relu(self.conv2(x))   # 6 x 25 x 25 -> 16 x 20 x 20
        x = self.pool(x)            # 16 x 20 x 20 -> 16 x 10 x 10
        x = x.view(-1, 16 * 10 * 10)  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Paper_Model(nn.Module):

    def __init__(self):
        """Model as defined in paper
        https://github.com/potterhsu/SVHNClassifier-PyTorch/blob/master/model.py
        """
        super(Paper_Model, self).__init__()

        hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48,
                      kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),  # in paper, the author used maxout units here
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64,
                      kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160,
                      kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192,
                      kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192,
                      kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192,
                      kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192,
                      kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden9 = nn.Sequential(
            nn.Linear(192 * 7 * 7, 3072),
            nn.ReLU()
        )
        hidden10 = nn.Sequential(
            nn.Linear(3072, 3072),
            nn.ReLU()
        )

        self._features = nn.Sequential(
            hidden1,
            hidden2,
            hidden3,
            hidden4,
            hidden5,
            hidden6,
            hidden7,
            hidden8
        )
        self._classifier = nn.Sequential(
            hidden9,
            hidden10
        )
        self._digit_length = nn.Sequential(nn.Linear(3072, 7))

    def forward(self, x):
        x = self._features(x)
        x = x.view(x.size(0), 192 * 7 * 7)
        x = self._classifier(x)
        x = self._digit_length(x)
        return x


def ChenXi(pretrained=True):
    """Model by Aaron Chen
    https://github.com/aaron-xichen/pytorch-playground/blob/master/svhn/model.py
    """
    return svhn(32, pretrained)
