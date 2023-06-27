import torch
from forward import (
    betas,
    sqrt_one_minus_alphas_cumprod,
    sqrt_recip_alphas,
    posterior_variance,
)

from forward import get_index_from_list
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from torchvision.utils import make_grid
from torchvision.utils import save_image


def show_tensor_image(image):
    reverse_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            transforms.Lambda(lambda t: t * 255.0),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ]
    )

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))


@torch.no_grad()
def sample_timestep(x, t, model):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


revers_transforms = transforms.Compose(
    [
        transforms.Lambda(lambda t: (t + 1) / 2),
    ]
)


@torch.no_grad()
def sample_plot_image(device="cpu", IMG_SIZE=28, T=300, model=None, epoch=0):
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    num_images = 10
    stepsize = int(T / num_images)
    image_list = []

    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t, model)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        img = revers_transforms(img)

        # print(img.shape)
        if i % stepsize == 0:
            # print(img.max(), img.min())
            image_list.append(img.reshape(3, img_size, img_size))

    # Create a grid of images
    image_grid = make_grid(
        image_list,
        nrow=5,
    )  # Adjust nrow as needed
    save_image(image_grid, f"sample_{epoch}.png")
