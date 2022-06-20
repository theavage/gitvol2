import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from IPython.display import clear_output

from utils import *
from unet import UNet
from losses import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

size = 128
transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])

batch_size = 6
dataset = ISIC(style = -1, transform=transform, type='train')
trainset, valset = torch.utils.data.random_split(dataset, [270, 30])
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=3)

model = UNet().to(device)

num_epochs = 10
criterion = bce_loss
optimizer = optim.Adam(model.parameters(), lr=0.00001)

for e in range(num_epochs):
    print('epoch %d/%d' % (e+1, num_epochs))
    model.train()
    average_train_loss = 0
    for X_batch, y_batch in trainloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_batch, y_pred)
        loss.backward()
        optimizer.step()
        average_train_loss += loss / len(trainloader)
    print(' - trainloss: %f' % average_train_loss)

    model.eval()
    average_val_loss = 0
    for X_batch, y_batch in valloader:
        y_pred = torch.sigmoid(X_batch).detach().cpu()
        y_pred = y_pred > 0.5
        loss = loss = criterion(y_batch, y_pred)
        average_val_loss += loss / len(trainloader)
    print(' - valloss: %f' % average_val_loss)