import os
import numpy as np
import glob
import PIL.Image as Image

# pip install torchsummary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary
import torch.optim as optim
from time import time

import matplotlib.pyplot as plt
from IPython.display import clear_output

def predict(model, data, device):
    model.eval()  # testing mode
    y_pred = [F.sigmoid(model(X_batch.to(device))) for X_batch, _ in data]
    return np.array(y_pred)