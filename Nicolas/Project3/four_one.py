import torch
from torch.utils.data import DataLoader

from utils import *
from model import UNet
from losses import *
from sklearn.metrics import jaccard_score as jsc
from sklearn.metrics import accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = UNet().to(device)
X, y_true = getData()
y_temp = y_true

n_iterations = 10

n_epochs = 20
batch_size = 16
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
criterion = bce_loss

predset = dataset(X, y_temp)
predloader = DataLoader(predset, batch_size=batch_size, shuffle=True, num_workers=3)
trainsize = int(X.size(dim=0) * 0.9)
valsize = X.size(dim=0) - trainsize
cont = 0
res = []
acc = []
for i in range(n_iterations):
    cont = cont + 1
    dataset_temp = dataset(X, y_temp)
    trainset, valset = torch.utils.data.random_split(dataset_temp, [trainsize, valsize])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=3)
    model = train4(model, optimizer, n_epochs, criterion, trainloader, valloader, device, cont)
    y_pred = predict4_2(model, predloader, device)
    pred = y_pred.reshape(-1,1)
    y = y_true.reshape(-1,1)

    pred[pred < 0.3] = 0
    pred[pred >= 0.3] = 1

    res.append(jsc(y, pred))
    acc.append(accuracy_score(y, pred))
    print(res, acc)
    y_temp = preprocess(y_true, y_pred)
    id = 1
    plt.figure()
    plt.imshow(np.rollaxis(y_pred[id].numpy(), 0, 3), cmap='gray')
    plt.savefig('pred' + str(cont) + '.png')
plt.figure()
plt.plot(res)
plt.show()
plt.savefig('res.png')
plt.figure()
plt.plot(acc)
plt.show()
plt.savefig('acc.png')


torch.save(model.state_dict(), '/zhome/df/9/164401/gitvol2/Colin/project3/model_four_one.pt')