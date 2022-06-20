import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim

from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
num_classes = 1
model.fc = nn.Linear(in_features=num_ftrs, out_features=num_classes)
model.to(device)

size = 224
transform = transforms.transforms.Compose([transforms.transforms.ToTensor(), transforms.transforms.Resize((size, size))])
batch_size = 32
num_epochs = 50
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

set = ISIC2(transform)
trainsize = int(set.__len__() * 0.9)
valsize = set.__len__() - trainsize
trainset, valset = torch.utils.data.random_split(set, [trainsize, valsize])
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=3)

for e in range(num_epochs):
    train_loss = []
    model.train()
    for X, y in trainloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(X)
        loss = criterion(outputs, y.long())
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    val_loss = []
    model.eval()
    for X, y in valloader:
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            output = model(X)
        loss = criterion(output, y)
        val_loss.append(loss.item())

    print('e: ' + (e + 1) + ' t: ' + train_loss + ' v: ' + val_loss)

torch.save(model.state_dict(), '/zhome/df/9/164401/gitvol2/Colin/project3/model_four_two.pt')