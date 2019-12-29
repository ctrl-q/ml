import torch
from problem3 import Generator, VAE
import os
from torchvision.utils import save_image


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
OUTDIR = os.path.join("samples", "qual3")
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

alphas = [round(0 + i * 0.1, 1) for i in range(11)]


def generate_gan():
    checkpoint = torch.load("gan-q3.tar", map_location=DEVICE)["g_state_dict"]
    G = Generator()
    G.load_state_dict(checkpoint)
    G.to(DEVICE).eval()

    z0 = torch.randn(1, 100, device=DEVICE)
    z1 = torch.randn_like(z0)
    with torch.no_grad():
        x0 = G(z0)
        x1 = G(z1)
        out_latent = torch.cat(
            tuple(
                G(a * z0 + (1 - a) * z1) for a in alphas
            )
        )
        save_image(out_latent, os.path.join(OUTDIR, "gan-latent.png"), nrow=len(alphas))
        out_data = torch.cat(
            tuple(
                a * x0 + (1 - a) * x1 for a in alphas
            )
        )
        save_image(out_data, os.path.join(OUTDIR, "gan-data.png"), nrow=len(alphas))


def generate_vae():
    checkpoint = torch.load("vae-q3.tar", map_location=DEVICE)["vae_state_dict"]
    G = VAE()
    G.load_state_dict(checkpoint)
    G.to(DEVICE).eval()

    z0 = torch.randn(1, 100, device=DEVICE)
    z1 = torch.randn_like(z0)
    with torch.no_grad():
        x0 = G.decoder(z0)
        x1 = G.decoder(z1)
        out_latent = torch.cat(
            tuple(
                G.decoder(a * z0 + (1 - a) * z1) for a in alphas
            )
        )
        save_image(out_latent, os.path.join(OUTDIR, "vae-latent.png"), nrow=len(alphas))
        out_data = torch.cat(
            tuple(
                a * x0 + (1 - a) * x1 for a in alphas
            )
        )
        save_image(out_data, os.path.join(OUTDIR, "vae-data.png"), nrow=len(alphas))

if __name__ == "__main__":
    generate_gan()
    generate_vae()
