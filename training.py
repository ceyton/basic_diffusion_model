from torch.optim import Adam
import torch
from tqdm import tqdm
from loss import get_loss

from model import SimpleUnet
import torch
from torchvision.datasets import LFWPeople
import torchvision
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from forward import forward_diffusion_sample
from params import BATCH_SIZE
from sampling import sample_plot_image
from params import T
from params import DEVICE
from params import EPOCHS
from params import IMG_SIZE

transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((128, 128))])
train_data = LFWPeople(root="./data", download=True, transform=transform)
train_loader = DataLoader(train_data[:1000], batch_size=BATCH_SIZE, shuffle=True)


model = SimpleUnet()

model.to(DEVICE)
optimizer = Adam(model.parameters(), lr=0.001)


for epoch in range(EPOCHS):
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for step, batch in loop:
        optimizer.zero_grad()

        t = torch.randint(0, T, (BATCH_SIZE,), device=DEVICE).long()
        loss = get_loss(model, batch[0], t, device=DEVICE)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 and step == 0:
            print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
            sample_plot_image(device=DEVICE, IMG_SIZE=IMG_SIZE, T=T, model=model)
