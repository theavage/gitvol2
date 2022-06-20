import torch
import glob
import PIL.Image as Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output
from scipy.ndimage import median_filter
import numpy as np
 
class ISIC(torch.utils.data.Dataset):
    def __init__(self, type, style=-1): #style = -1 for not relevant
        'Initializations'

        if type == 'train':
            if style == -1:
                image_folder = '/dtu/datasets1/02514/isic/train_allstyles/Images/*.jpg'
                segmentation_folder = '/dtu/datasets1/02514/isic/train_allstyles/Segmentations/*.png'
            if style == 0:
                image_folder = '/dtu/datasets1/02514/isic/train_style0/Images/*.jpg'
                segmentation_folder = '/dtu/datasets1/02514/isic/train_style0/Segmentations/*.png'
            if style == 1:
                image_folder = '/dtu/datasets1/02514/isic/train_style1/Images/*.jpg'
                segmentation_folder = '/dtu/datasets1/02514/isic/train_style1/Segmentations/*.png'
            if style == 2:
                image_folder = '/dtu/datasets1/02514/isic/train_style2/Images/*.jpg'
                segmentation_folder = '/dtu/datasets1/02514/isic/train_style2/Segmentations/*.png'
        elif type == 'test':
            image_folder = '/dtu/datasets1/02514/isic/test_style0/Images/*.jpg'
            segmentation_folder = '/dtu/datasets1/02514/isic/test_style0/Segmentations/*.png'
        else:
            raise AttributeError()

        image_paths = sorted(glob.glob(image_folder))
        self.segmentation_paths = sorted(glob.glob(segmentation_folder))

        cropped_paths =[]

        for img in self.segmentation_paths: 
            base_img= os.path.split(img)[1][:12]
            cropped_paths.append(base_img)
            if cropped_paths.count(base_img) > 1:
                if style != -1 and type == 'train':
                    image_paths.append(str(image_paths[0][:45])+'/'+str(base_img)+'.jpg')
                    image_paths.sort()
                elif type == 'test':
                    image_paths.append(str(image_paths[0][:45])+str(base_img)+'.jpg')
                    image_paths.sort()
        
        self.final_image_paths = image_paths
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        'Returns the total number of samples'
        return len(self.final_image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image = Image.open(self.final_image_paths[idx])
        segmentation = Image.open(self.segmentation_paths[idx])
        X = self.transform(image)
        y = self.transform(segmentation)
        
        return X,y

def checkDevice():
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loadTestData(batch_size):
    
    testset = ISIC('test', -1)

    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=3)

    return test_loader

def loadTrainData(style,batch_size):
    '''
    Style = -1 gives allstyles
    '''

    dataset = ISIC('train', style)

    val_part = len(dataset)//4
    split = [len(dataset)-val_part,val_part]
    trainset,valset = random_split(dataset,split)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=3)

    return train_loader, val_loader

'''
How to get  data:
'''

trainloader,valloader = loadTrainData(0,24)

X_train, y_train = next(iter(trainloader))#this gets one batch
X_val, y_val = next(iter(valloader))

test_loader = loadTestData(32)
X_test,y_test = next(iter(test_loader))





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

def train4(model, opt, epochs, criterion, trainloader,valloader, device, cont):
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
        y_hat.numpy()
        y_hat[y_hat < 0.3] = 0
        y_hat[y_hat >= 0.3] = 1
        clear_output(wait=True)
        for k in range(6):
            plt.subplot(2, 6, k+1)
            plt.imshow(np.rollaxis(X_test[k].numpy(), 0, 3), cmap='gray')
            plt.title('Real')
            plt.axis('off')

            plt.subplot(2, 6, k+7)
            plt.imshow(y_hat[k, 0], cmap='gray')
            plt.title('Output')
            plt.axis('off')

        #plt.title('Predictions using annotations with style 1')
        plt.suptitle('%d / %d - loss: %f' % (epoch+1, epochs, avg_loss))
        plt.savefig('mask_' + str(cont) + '.png')

    return model

def predict4(model, data, device):
    model.eval()
    model.to(device)
    return torch.sigmoid(model(data.to(device)))

def predict4_2(model, data, device):
    model.eval()
    res = torch.empty((0, 1, 256, 256))
    for X_batch, _ in data:
        X_batch = X_batch.to(device)
        y_pred = torch.sigmoid(model(X_batch))
        res = torch.cat([res, y_pred.detach().cpu()], dim=0)
        res = res > 0.5
    return res