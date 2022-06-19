import torch
import numpy as np
import os
import glob


from model import *
from utils import *
from train import * 
from predict import * 
from new_unet import *

PREDICT = False

device = checkDevice()

model = new_UNet().to(device)

if PREDICT == False:
    trainloader,valloader = loadTrainData(0,12)
    #X_train, y_train = next(iter(trainloader))
    #X_val, y_val = next(iter(valloader))

    model = train(model, optim.Adam(model.parameters(),lr=0.0001), 50, trainloader,valloader, device)

    path = 'model_style0.pt'
    torch.save(model.state_dict(), path)    

else:

    path = 'model_style0.pt'
    model.load_state_dict(torch.load(path))

    device = checkDevice()

    testloader= loadTestData(6)
    X_test_loader, y_test_loader = next(iter(testloader))
    
    pred = predict(model, testloader, device) 
    #res = evaluate(pred, y_test_loader)


