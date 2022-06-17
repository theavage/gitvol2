import torch
import numpy as np
import os
import glob


from model import *
from utils import *
from train import *

data_path_X = '/dtu/datasets1/02514/isic/train_allstyles/Images/*.jpg'
data_path_y = '/dtu/datasets1/02514/isic/train_allstyles/Segmentations/*.png'

X_train_loader, X_val_loader, y_train_loader, y_val_loader = loadData(data_path_X, data_path_y, 32)
device = checkDevice()

model = UNet().to(device)

train(model, optim.Adam(model.parameters()), 20, X_train_loader, X_val_loader, y_train_loader, y_val_loader, device)


