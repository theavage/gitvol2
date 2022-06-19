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
from sklearn.metrics import jaccard_score as jsc
from sklearn.metrics import accuracy_score



def predict(model, data, device):
    #model.to(device)
    model.eval()  # testing mode
    res = np.empty((0, 1, 256, 256))
    for X_batch, _ in data:
        X_batch = X_batch.to(device)
        y_pred = F.sigmoid(model(X_batch))
        #torch.save(y_pred, 'tensor.pt')
        res = np.concatenate((res, y_pred.detach().cpu().numpy()), axis=0)
    return res

def evaluate(pred, y):
    pred = pred.reshape(-1,1)
    y = y.reshape(-1,1)

    pred[pred < 0.3] = 0
    pred[pred >= 0.3] = 1

    res = jsc(y, pred)
    acc = accuracy_score(y, pred)

    return res, acc

