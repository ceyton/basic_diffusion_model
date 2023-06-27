import torch


BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T = 300
IMG_SIZE = 128
EPOCHS = 100
