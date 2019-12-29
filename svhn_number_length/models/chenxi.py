import os
from collections import OrderedDict

import torch
import torch.nn as nn

model_urls = {
    'svhn': 'https://web.archive.org/web/20180115042531if_/http://ml.cs.tsinghua.edu.cn:80/~chenxi/pytorch-models/svhn-f564f3d8.pth',
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))  # this file's location


class ChenXi(nn.Module):
    def __init__(self, features, n_channel, num_classes):
        super(ChenXi, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(n_channel, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=padding)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels,
                                                  affine=False), nn.ReLU(), nn.Dropout(0.3)]
            else:
                layers += [conv2d, nn.ReLU(), nn.Dropout(0.3)]
            in_channels = out_channels
    return nn.Sequential(*layers)


def svhn(n_channel, pretrained):
    cfg = [n_channel, n_channel, 'M', 2 * n_channel, 2 * n_channel,
           'M', 4 * n_channel, 4 * n_channel, 'M', (8 * n_channel, 0), 'M']
    layers = make_layers(cfg, batch_norm=True)
    model = ChenXi(layers, n_channel=8 * n_channel, num_classes=10)
    if pretrained:
        m = torch.load(os.path.join(HERE, "chenxi.pth"), map_location=DEVICE)
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    # Reshape last layer for our purposes
    model.classifier = nn.Sequential(nn.Linear(1024, 7))
    return model
