import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from IPython import display
import matplotlib.pylab as plt
import ipywidgets
from PIL import Image

if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 64
trainset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(

            nn.Linear(100, 2848),
            nn.BatchNorm1d(2848),
            nn.LeakyReLU(True),

            nn.Linear(2848, 2848),
            nn.BatchNorm1d(2848),
            nn.LeakyReLU(True),

            nn.Linear(2848, 2848),
            nn.BatchNorm1d(2848),
            nn.LeakyReLU(True),

            nn.Linear(2848, 28*28),
            nn.BatchNorm1d(28*28),
            nn.LeakyReLU(True),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), 1, 28, 28)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(

            nn.Linear(28*28, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(),

            nn.Linear(256, 1)

        )
    
    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        return x

    #Initialize networks
d = Discriminator().to(device)
g = Generator().to(device)
d_opt = torch.optim.Adam(d.parameters(), 0.0004, (0.5, 0.999))
g_opt = torch.optim.Adam(g.parameters(), 0.0001, (0.5, 0.999))

plt.figure(figsize=(20,10))
subplots = [plt.subplot(2, 6, k+1) for k in range(12)]
num_epochs = 100
discriminator_final_layer = torch.sigmoid


for epoch in range(num_epochs):
    for minibatch_no, (x, target) in enumerate(train_loader):
        x_real = x.to(device)*2-1 #scale to (-1, 1) range
        z = torch.randn(x.shape[0], 100).to(device)
        x_fake = g(z)

        #Update discriminator
        d.zero_grad()

        output = d(x_real)
        loss = nn.CrossEntropyLoss()
        real_loss = loss(output,torch.ones_like(output))

        output = d(x_fake.detach())
        
        fake_loss = loss(output,torch.zeros_like(output))
        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_opt.step()

        #Update generator
        g.zero_grad()
        output = d(x_fake)
        g_loss = loss(output,torch.ones_like(output))
        g_loss.backward()
        g_opt.step()

        assert(not np.isnan(d_loss.item()))
        #Plot results every 100 minibatches
        if minibatch_no % 100 == 0:
            with torch.no_grad():
                P = discriminator_final_layer(d(x_fake))
                for k in range(11):
                    x_fake_k = x_fake[k].cpu().squeeze()/2+.5
                    subplots[k].imshow(x_fake_k, cmap='gray')
                    #subplots[k].set_title('d(x)=%.2f' % P[k])
                    subplots[k].axis('off')
                    
                z = torch.randn(batch_size, 100).to(device)
                H1 = discriminator_final_layer(d(g(z))).cpu()
                H2 = discriminator_final_layer(d(x_real)).cpu()
                plot_min = min(H1.min(), H2.min()).item()
                plot_max = max(H1.max(), H2.max()).item()
                subplots[-1].cla()
                subplots[-1].hist(H1.squeeze(), label='fake', range=(plot_min, plot_max), alpha=0.5)
                subplots[-1].hist(H2.squeeze(), label='real', range=(plot_min, plot_max), alpha=0.5)
                subplots[-1].legend()
                subplots[-1].set_xlabel('Probability of being real')
                subplots[-1].set_title('Discriminator loss: %.2f' % d_loss.item())
                
                
                title = 'Epoch {e} - minibatch {n}/{d}'.format(e=epoch+1, n=minibatch_no, d=len(train_loader))
                plt.gcf().suptitle(title, fontsize=20)
                display.display(plt.gcf())
                plt.savefig('/zhome/63/0/173423/02514/gitvol2/Thea/plot.png')
                display.clear_output(wait=True)
                
