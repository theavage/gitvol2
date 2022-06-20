import torch
import glob
from PIL import Image
from torchvision import transforms
from scipy.ndimage import median_filter
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os

def getData():
    image_folder = '/dtu/datasets1/02514/isic/train_style2/Images/*.jpg'
    segmentation_folder = '/dtu/datasets1/02514/isic/train_style2/Segmentations/*.png'

    image_paths = sorted(glob.glob(image_folder))
    segmentation_paths = sorted(glob.glob(segmentation_folder))

    cropped_paths =[]

    for img in segmentation_paths: 
        base_img= os.path.split(img)[1][:12]
        cropped_paths.append(base_img)
        if cropped_paths.count(base_img) > 1:
            image_paths.append(str(image_paths[0][:45])+'/'+str(base_img)+'.jpg')
    image_paths.sort()
    
    transform = transforms.ToTensor()

    X = torch.empty((0, 3, 256, 256))
    y = torch.empty((0))

    for i, s in zip(image_paths, segmentation_paths):
        image = Image.open(i)
        segmentation = Image.open(s)
        X = torch.cat([X, transform(image).unsqueeze(dim=0)], dim=0)
        y = torch.cat([y, transform(segmentation).unsqueeze(dim=0)], dim=0)
    
    return X, y

class dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.data = X
        self.target = y

    def __len__(self):
        return self.data.size(dim=0)

    def __getitem__(self, idx):
        return self.data[idx, :, :, :], self.target[idx, :, :, :]

def preprocess(y_true, y_pred):
    # step 1
    y_new = y_true * y_pred

    # step 2
    ious = torch.count_nonzero(torch.logical_and(y_true, y_pred), dim=(1, 2, 3)) \
        / torch.count_nonzero(torch.logical_or(y_true, y_pred), dim=(1, 2, 3))
    mask = ious < 0.5
    y_new[mask, :, :, :] = y_true[mask, :, :, :]

    # step 3
    y_new = torch.as_tensor(median_filter(y_new, size=5))

    return y_new

def train(model, opt, epochs, criterion, trainloader,valloader, device):
    X_test, y_test = next(iter(valloader))

    for epoch in range(epochs):
        print('* Epoch %d/%d' % (epoch+1, epochs))

        avg_loss = 0
        model.train()  # train mode
        for X_batch, y_batch in trainloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # set parameter gradients to zero
            opt.zero_grad()

            # forward
            y_pred = model(X_batch)
            loss = criterion(y_batch, y_pred)  # forward-pass
            loss.backward()  # backward-pass
            opt.step()  # update weights

            # calculate metrics to show the user
            avg_loss += loss / len(trainloader)
        print(' - train loss: %f' % avg_loss)

        # show intermediate results
        model.eval()  # testing mode
        y_hat = torch.sigmoid(model(X_test.to(device))).detach().cpu()
        # val_loss = criterion(y_hat, y_test) / len(trainloader)
        # print(' - val loss: %f' % val_loss)
        # clear_output(wait=True)
        # for k in range(6):
        #     plt.subplot(2, 6, k+1)
        #     plt.imshow(np.rollaxis(X_test[k].numpy(), 0, 3), cmap='gray')
        #     plt.title('Real')
        #     plt.axis('off')

        #     plt.subplot(2, 6, k+7)
        #     plt.imshow(y_hat[k, 0], cmap='gray')
        #     plt.title('Output')
        #     plt.axis('off')

        # plt.title('Predictions using annotations with style 1')
        # plt.suptitle('%d / %d - loss: %f' % (epoch+1, epochs, avg_loss))
        # plt.savefig('style1.png')

    return model

def predict(model, data, device):
    model.eval()
    model.to(device)
    return torch.sigmoid(model(data.to(device)))

def predict2(model, data, device):
    model.eval()
    res = torch.empty((0, 1, 256, 256))
    for X_batch, _ in data:
        X_batch = X_batch.to(device)
        y_pred = torch.sigmoid(model(X_batch))
        res = torch.cat([res, y_pred.detach().cpu()], dim=0)
        res = res > 0.5
    return res

class ISIC2(torch.utils.data.Dataset):
    def __init__(self, transform):
        self.transform = transform
        folder_background = '/dtu/datasets1/02514/isic/background/*.jpg'
        folder_objects = '/dtu/datasets1/02514/isic/train_allstyles/Images/*.jpg'
        paths_background = sorted(glob.glob(folder_background))
        paths_object = sorted(glob.glob(folder_objects))
        labels_background = torch.zeros((len(paths_background)))
        labels_objects = torch.ones((len(paths_object)))
        self.image_paths = paths_background + paths_object
        self.labels = torch.cat([labels_background, labels_objects])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = self.labels[idx]
        return self.transform(image), label