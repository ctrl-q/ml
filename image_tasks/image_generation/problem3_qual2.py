import torch
from problem3 import Generator, VAE
import os
from torchvision.utils import save_image


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
OUTDIR = os.path.join("samples", "qual2")
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def generate_gan():
    checkpoint = torch.load("gan-q3.tar", map_location=DEVICE)["g_state_dict"]
    G = Generator()
    G.load_state_dict(checkpoint)
    G.to(DEVICE).eval()

    z = torch.randn(1, 100, device=DEVICE)
    outdir = os.path.join(OUTDIR, "gan")
    os.makedirs(outdir, exist_ok=True)
    generated = G(z)
    save_image(generated, os.path.join(outdir, "original.png"), normalize=True)

    eps = 10
    with torch.no_grad():
        for i in range(100):
            z[0, i] += eps
            perturbed = G(z)
            save_image(perturbed, os.path.join(outdir, str(i) + ".png"), normalize=True)
            z[0, i] -= eps


def generate_vae():
    checkpoint = torch.load("vae-q3.tar", map_location=DEVICE)["vae_state_dict"]
    G = VAE()
    G.load_state_dict(checkpoint)
    G.to(DEVICE).eval()

    z = torch.randn(1, 100, device=DEVICE)
    outdir = os.path.join(OUTDIR, "vae")
    os.makedirs(outdir, exist_ok=True)
    generated = G.decoder(z)
    save_image(generated, os.path.join(outdir, "original.png"), normalize=True)

    eps = 10
    with torch.no_grad():
        for i in range(100):
            z[0, i] += eps
            perturbed = G.decoder(z)
            save_image(perturbed, os.path.join(outdir, str(i) + ".png"), normalize=True)
            z[0, i] -= eps

if __name__ == "__main__":
    generate_gan()
    generate_vae()