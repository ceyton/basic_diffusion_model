from forward import forward_diffusion_sample
import torch.nn.functional as F


def get_loss(model, x_0, t, device="cpu"):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)
