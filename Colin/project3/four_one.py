import torch
from torch.utils.data import DataLoader

from utils import *
from unet import UNet
from losses import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = UNet().to(device)
X, y_true = getData()
y_temp = y_true

n_iterations = 10

n_epochs = 50
batch_size = 16
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
criterion = bce_loss

predset = dataset(X, y_temp)
predloader = DataLoader(predset, batch_size=batch_size, shuffle=True, num_workers=3)
trainsize = int(X.size(dim=0) * 0.9)
valsize = X.size(dim=0) - trainsize

for i in range(n_iterations):
    
    dataset_temp = dataset(X, y_temp)
    trainset, valset = torch.utils.data.random_split(dataset_temp, [trainsize, valsize])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=3)
    model = train(model, optimizer, n_epochs, criterion, trainloader, valloader, device)
    y_pred = predict2(model, predloader, device)
    y_temp = preprocess(y_true, y_pred)

torch.save(model.state_dict(), '/zhome/df/9/164401/gitvol2/Colin/project3/model_four_one.pt')