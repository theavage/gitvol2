import torch
import glob
import PIL.Image as Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

class ISIC(torch.utils.data.Dataset):
    def __init__(self, train, transform, data_path):
        'Initialization'
        if train == True:
            self.transform = transform
            self.image_paths = sorted(glob.glob(data_path)*3)
        else:
            self.transform = transform
            self.image_paths = sorted(glob.glob(data_path))
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]

        
        image = Image.open(image_path)
        X = self.transform(image)
        return X


def checkDevice():
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def loadTrainData(data_path_X, data_path_y, batch_size):
    


    transform = transforms.Compose([transforms.ToTensor()])

    Xset = ISIC(True, transform=transform, data_path=data_path_X)
    yset = ISIC(False, transform=transform, data_path=data_path_y)

    X_train, X_val = torch.utils.data.random_split(Xset, [270, 30])
    y_train, y_val = torch.utils.data.random_split(yset, [270, 30])

    X_train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True, num_workers=3)
    X_val_loader = DataLoader(X_val, batch_size=batch_size, shuffle=True, num_workers=3)
    y_train_loader = DataLoader(y_train, batch_size=batch_size, shuffle=False, num_workers=3)
    y_val_loader = DataLoader(y_val, batch_size=batch_size, shuffle=False, num_workers=3)


    return X_train_loader, X_val_loader, y_train_loader, y_val_loader


def loadTestData(data_path_X, data_path_y, batch_size):
    


    transform = transforms.Compose([transforms.ToTensor()])

    Xset = ISIC(False, transform=transform, data_path=data_path_X)
    yset = ISIC(False, transform=transform, data_path=data_path_y)

    print(len(Xset.image_paths))
    print(len(yset.image_paths))

    X_test_loader = DataLoader(Xset, batch_size=batch_size, shuffle=True, num_workers=3)
    y_test_loader = DataLoader(yset, batch_size=batch_size, shuffle=False, num_workers=3)


    return X_test_loader, y_test_loader