import torch
import numpy as np
import os
import glob


from model import *
from utils import *
from train import * 
from predict import * 

PREDICT = False

data_path_train_X = '/dtu/datasets1/02514/isic/train_allstyles/Images/*.jpg'
data_path_train_y = '/dtu/datasets1/02514/isic/train_allstyles/Segmentations/*.png'

data_path_test_X = '/dtu/datasets1/02514/isic/test_style0/Images/*.jpg'
data_path_test_y = '/dtu/datasets1/02514/isic/test_style0/Segmentations/*.png'


device = checkDevice()

model = UNet().to(device)

if PREDICT == False:
    X_train_loader, X_val_loader, y_train_loader, y_val_loader = loadTrainData(data_path_train_X, data_path_train_y, 32)

    train(model, optim.Adam(model.parameters()), 20, X_train_loader, X_val_loader, y_train_loader, y_val_loader, device)

else:
    X_test_loader, y_test_loader = loadTestData(data_path_test_X, data_path_test_y, 32)
    pred = predict(model, X_test_loader, device) 


