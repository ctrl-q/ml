# Code for Training VAE and GAN, for use in two other scripts

import os

import torch
from torch.autograd import grad
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torchvision.utils import save_image

from classify_svhn import get_data_loader

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

SAMPLES_DIR = os.path.join("samples")
for model in ("gan", "vae"):
    os.makedirs(os.path.join(SAMPLES_DIR, model), exist_ok=True)

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class Squeeze(nn.Module):
    def forward(self, inputs):
        return inputs.squeeze()


class Unsqueeze(nn.Module):
    def forward(self, inputs):
        return inputs.view(-1, 256, 1, 1)


class Interpolate(nn.Module):
    def forward(self, inputs):
        return F.interpolate(inputs, scale_factor=2, mode='bilinear', align_corners=True)


# Common architecture for both models
d = lambda: nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.ELU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(32, 64, 5),
            nn.ELU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(64, 256, 5),
            nn.ELU(),
            Squeeze()
)

g = lambda: nn.Sequential(
            nn.Linear(100, 256),
            Unsqueeze(),
            nn.ELU(),
            nn.Conv2d(256, 64, 5, 1, 4),
            nn.ELU(),
            Interpolate(),
            nn.Conv2d(64, 32, 3, 1, 2),
            nn.ELU(),
            Interpolate(),
            nn.Conv2d(32, 16, 3, 1, 3),
            nn.ELU(),
            nn.Conv2d(16, 3, 3, 1, 3)
)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            *g(),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        out = self.model(inputs)
        return out.view(-1, 3, 32, 32)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            *d(),
            nn.Linear(256, 1)
        )

    def forward(self, inputs):
        return self.model(inputs)


def gradient_penalty(Critic, x_p, x_q, lamda=10):
    alpha = torch.rand(x_p.size(0), 1, 1, 1, device=x_p.device)
    inputs = (alpha * x_p + ((1 - alpha) * x_q))
    inputs.requires_grad = True
    outputs = Critic(inputs)
    noise = torch.ones(x_p.size(0), 1, device=x_p.device)

    gradients, *_ = grad(outputs, inputs, noise, True)
    gradients = gradients.view(gradients.size(0), -1)
    penalty = lamda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty


def ELBO(inputs, outputs, mu, log_sigma):
    loss = nn.MSELoss(reduction="sum")
    return loss(inputs, outputs) - (1 + log_sigma - mu.pow(2) - log_sigma.exp()).sum() / 2


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            *d(),
            nn.Linear(256, 200)
        )

        self.decoder = g()

    def forward(self, x):
        q_params = self.encoder(x)
        mu = q_params[:, 100:]
        log_sigma = q_params[:, :100]
        z = self.reparameterize(mu, log_sigma)
        return self.decoder(z), mu, log_sigma

    def reparameterize(self, mu, log_sigma):
        sigma = (log_sigma / 2).exp()
        e = torch.randn_like(sigma)
        return mu + sigma * e


def train(D, G, dataloader, num_epochs=120):
    return train_vae(D, dataloader, num_epochs) if isinstance(D, VAE) else train_gan(D, G, dataloader, num_epochs)


def train_vae(vae, dataloader, num_epochs=120):
    vae.to(DEVICE).train()
    adam = Adam(vae.parameters(), lr=0.0005, betas=(0.5, 0.999))
    try:
        for epoch in range(num_epochs):
            epoch_loss = 0
            for i, (inputs, _) in enumerate(dataloader):
                inputs = inputs.to(DEVICE)
                adam.zero_grad()
                outputs, mu, log_sigma = vae(inputs)
                loss = ELBO(inputs, outputs, mu, log_sigma)
                loss.backward()
                epoch_loss += loss.item() / inputs.size(0)
                adam.step()
            print("Epoch:", epoch, "Loss:", epoch_loss / i)

            with torch.no_grad():
                z = torch.randn(inputs.size(0), 100, device=DEVICE)
                generated = vae.decoder(z)
                save_image(generated, os.path.join(SAMPLES_DIR, "vae", str(epoch) + ".png"), normalize=True, nrow=2)
    finally:
        checkpoint = {
            "vae_state_dict": vae.state_dict(),
            "adam_state_dict": adam.state_dict(),
            "epoch": epoch,
            "loss": loss,
        }
        torch.save(checkpoint, "vae-q3.tar")


def train_gan(D, G, dataloader, num_epochs=120):
    G.to(DEVICE).train()
    D.to(DEVICE).train()

    adam_g = Adam(G.parameters(), lr=0.0005, betas=(
        0.5, 0.999))
    adam_d = Adam(D.parameters(), lr=0.0005, betas=(0.5, 0.999))

    try:
        for epoch in range(num_epochs):
            for i, (inputs, _) in enumerate(dataloader):
                inputs = inputs.to(DEVICE)
                adam_d.zero_grad()
                z = torch.randn(inputs.size(0), 100, device=DEVICE)
                outputs = G(z)

                inputs_score = D(inputs)
                outputs_score = D(outputs)
                penalty = gradient_penalty(D, inputs.data, outputs.data)
                loss_d = outputs_score.mean() - inputs_score.mean() + penalty

                loss_d.backward()
                adam_g.zero_grad()
                adam_d.step()

                if i > 0 and i % 5 == 0:  # train G every 5 mini-batches
                    generated = G(z)
                    generated_score = D(generated)
                    loss_g = -generated_score.mean()
                    loss_g.backward()
                    adam_g.step()

            print("Epoch:", epoch, "; Discriminator loss:",
                  loss_d.item(), "; Generator loss:", loss_g.item())
            save_image(generated, os.path.join(
                SAMPLES_DIR, "gan-norm", str(epoch) + ".png"), normalize=True, nrow=2)
    finally:
        checkpoint = {
            "d_state_dict": D.state_dict(),
            "g_state_dict": G.state_dict(),
            "adam_d_state_dict": adam_d.state_dict(),
            "adam_g_state_dict": adam_g.state_dict(),
            "epoch": epoch,
            "loss_d": loss_d,
            "loss_g": loss_g
        }
        torch.save(checkpoint, "gan-q3-norm.tar")


if __name__ == "__main__":
    dataloader, *_ = get_data_loader("svhn", 32)
    train(Discriminator(), Generator(), dataloader)
    train(VAE(), None, dataloader, 3)
