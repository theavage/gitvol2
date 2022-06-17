import torch
import glob
import PIL.Image as Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

class ISIC(torch.utils.data.Dataset):
    def __init__(self, transform, data_path):
        'Initialization'
        self.transform = transform
        self.image_paths = sorted(glob.glob(data_path))
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        image = Image.open(image_path)
        label = Image.open(label_path)
        y = self.transform(label)
        X = self.transform(image)
        return X, y


def checkDevice():
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def loadData(data_path_X, data_path_y, batch_size):
    


    transform = transforms.Compose([transforms.ToTensor()])

    Xset = ISIC(transform=transform, data_path=data_path_X)
    yset = ISIC(transform=transform, data_path=data_path_y)

    print('Loaded %d training images' % len(Xset))
    print(Xset[0])
    print(Xset[0].shape)
    print('Loaded %d test images' % len(yset))

    X_train, X_val = torch.utils.data.random_split(Xset, [90, 10])
    y_train, y_val = torch.utils.data.random_split(yset, [90, 10])

    X_train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True, num_workers=3)
    X_val_loader = DataLoader(X_val, batch_size=batch_size, shuffle=True, num_workers=3)
    y_train_loader = DataLoader(y_train, batch_size=batch_size, shuffle=False, num_workers=3)
    y_val_loader = DataLoader(y_val, batch_size=batch_size, shuffle=False, num_workers=3)


    return X_train_loader, X_val_loader, y_train_loader, y_val_loader