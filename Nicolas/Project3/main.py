import torch
import numpy as np
import os
import glob


from model import *
from utils import *
from train import * 
from predict import * 
#from new_unet import *

PREDICT = True

device = checkDevice()

model = UNet().to(device)

if PREDICT == False:
    trainloader,valloader = loadTrainData(-1,12)
    #X_train, y_train = next(iter(trainloader))
    #X_val, y_val = next(iter(valloader))

    model = train(model, optim.Adam(model.parameters(),lr=0.0001), 5, trainloader,valloader, device)

    path = 'model.pt'
    torch.save(model.state_dict(), path)   

else:

    path = 'model.pt'
    model.load_state_dict(torch.load(path))

    testloader= loadTestData(6)
    X_test_loader, y_test_loader = next(iter(testloader))
    
    pred = predict(model, testloader, device) 

    y_iterator = iter(testloader)
    labels = np.empty((0, 1, 256, 256))
    for i in range(len(testloader)):
        _, label = next(y_iterator)
        labels = np.concatenate((labels, label), axis=0)

    res, acc = evaluate(pred, labels)
    print(res, acc)
    #res = iou_pytorch(pred, labels)


