# -*- coding: utf-8 -*-
# Variational AutoEncoder trained on Binarized MNIST and evaluated with ELBO

# References:
# https://github.com/pytorch/examples/blob/master/vae/main.py

import numpy as np
import torch
import torch.utils.data as data_utils
import os

from torch import nn, optim
from torchvision.datasets import utils
from torch.functional import F
from scipy import stats
from torch.distributions.normal import Normal

### Data Loading ###


def get_data_loader(dataset_location, batch_size):
    URL = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/"
    # start processing

    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])
    splitdata = []
    for splitname in ["train", "valid", "test"]:
        filename = "binarized_mnist_%s.amat" % splitname
        filepath = os.path.join(dataset_location, filename)
        utils.download_url(URL + filename, dataset_location)
        with open(filepath) as f:
            lines = f.readlines()
        x = lines_to_np_array(lines).astype('float32')
        x = x.reshape(x.shape[0], 1, 28, 28)
        # pytorch data loader
        data_utils.TensorDataset(torch.from_numpy(x))
        dataset_loader = data_utils.DataLoader(
            x, batch_size=batch_size, shuffle=splitname == "train")
        splitdata.append(dataset_loader)
    return splitdata


os.makedirs("mnist", exist_ok=True)
trainset, validset, testset = get_data_loader("mnist", batch_size=64)


class VAE(nn.Module):
    def __init__(self, latent_dimension=100):
        super(VAE, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=5)
        self.conv4 = nn.Conv2d(256, 64, kernel_size=5, padding=4)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=2)
        self.conv6 = nn.Conv2d(32, 16, kernel_size=3, padding=2)
        self.conv7 = nn.Conv2d(16, 1, kernel_size=3, padding=2)

        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)

        self.fc_log_sigma = nn.Linear(256, 2 * latent_dimension)
        self.fc_decoder = nn.Linear(latent_dimension, 256)

    def encode(self, x):

        x = F.elu(self.conv1(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        x = F.elu(self.conv2(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        x = F.elu(self.conv3(x))
        x = x.reshape(-1, 256)

        x = self.fc_log_sigma(x)

        return x[:, :latent_dimension], x[:, latent_dimension:]

    def reparam(self, mu, log_sigma):

        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)

        return mu + eps * std

    def decode(self, z):

        x = F.elu(self.fc_decoder(z))

        x = x.reshape(z.shape[0], 256, 1, 1)

        x = F.elu(self.conv4(x))
        x = self.upsampling(x)

        x = F.elu(self.conv5(x))
        x = self.upsampling(x)

        x = F.elu(self.conv6(x))
        x = self.conv7(x)

        return torch.sigmoid(x)

    def forward(self, x):
        mu, log_sigma = self.encode(x)
        z = self.reparam(mu, log_sigma)
        x_tild = self.decode(z)
        return x_tild, mu, log_sigma

###Train VAE###


torch.manual_seed(23)
device = torch.device("cpu")
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)
n_epochs = 20
batch_size = 64
latent_dimension = 100


def loss_function(x_tild, x, mu, log_sigma):
    x = x.view(-1, 784)
    x_tild = x_tild.view(-1, 784)
    Bce = F.binary_cross_entropy(x_tild, x, reduction='sum')
    KL_D = 0.5 * torch.sum(1 + log_sigma - mu**2 - log_sigma.exp())
    return Bce - KL_D


def ELBO_eval(data_loader):
    with torch.no_grad():
        model.eval()
        loss = 0
        minibatches = len(data_loader.dataset)
        for i, data in enumerate(data_loader):
            data = data.to(device)
            recon_batch, mu, log_sigma = model(data)
            loss += loss_function(recon_batch, data, mu, log_sigma).item()

    return -loss / minibatches


def train(epoch):
    model.train()
    train_loss = 0
    for i, data in enumerate(trainset):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_sigma = model(data)
        loss = loss_function(recon_batch, data, mu, log_sigma)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if i % 100 == 0:
            print("Epoch:", epoch, "Step:", i * len(data), "/",
                  len(trainset.dataset), "Loss:", loss.item() / len(data))

    print("End of epoch:", epoch, "Loss:", train_loss / len(trainset.dataset))


if __name__ == "__main__":
    for epoch in range(1, n_epochs + 1):
        train(epoch)

ELBO_valid = ELBO_eval(validset)
ELBO_test = ELBO_eval(testset)
print('ELBO:', 'Validation', ELBO_valid, "Test:", ELBO_valid)

### Importance Sampling ###


def importance_sampling(model, data, samples):

    K = samples.size(1)
    mu, log_sigma = model.encode(data)
    sig = log_sigma.exp().sqrt()
    logs = []

    for i, (x, z, m, s) in enumerate(
            zip(data, samples, mu, sig)):  # iterate through a batch

        log_q_z_given_x = Normal(m, s).log_prob(z).sum(1).float()

        p_z = stats.norm.pdf(z)
        log_p_z = torch.tensor(p_z).log().sum(1).float()

        x_hat = model.decode(z.view(K, 1, -1))
        min_ = torch.finfo(torch.float64).eps
        max_ = 1 - min_
        x_hat = torch.clamp(x_hat, min_, max_)
        log_p_x_given_z = torch.sum(
            (x * (x_hat).log() + (1 - x) * (1 - x_hat).log()).view(K, -1), 1).float()

        terms = log_p_x_given_z + log_p_z - log_q_z_given_x
        log_p_x = terms.logsumexp(0) - torch.log(torch.tensor(K).float())

        logs.append(log_p_x)

    return logs


def gen_samples(K, mu, sig):
    return np.array([stats.norm.rvs(loc=mi, scale=si, size=K, random_state=None)
                     for mi, si in zip(mu, sig)]).transpose()


def importance_sampling_eval(dataloader, K):

    logs = []
    for inputs in dataloader:
        inputs = inputs.to(device)
        mu, log_sigma = model.encode(inputs)
        sig = log_sigma.exp().sqrt()

        samples = torch.Tensor([gen_samples(K, mi, si) for (
            mi, si) in zip(mu.data.numpy(), sig.data.numpy())])
        logs_inputs = importance_sampling(model, inputs, samples)
        logs.extend(logs_inputs)

    return logs


data = next(iter(validset))

mu, log_sigma = model.encode(data)
sig = torch.sqrt(torch.exp(log_sigma))

samples = torch.Tensor([gen_samples(200, mi, si) for (
    mi, si) in zip(mu.data.numpy(), sig.data.numpy())])  # TODO

logs = importance_sampling(model, data, samples)
print(torch.tensor(logs).mean().item())

log_p_x_valid = importance_sampling_eval(validset, 200)
print('Validation log-likelihood:', torch.Tensor(log_p_x_valid).mean().item())

log_p_x_test = importance_sampling_eval(testset, 200)
print('Test log-likelihood', torch.Tensor(importance_sampling).mean().item())
