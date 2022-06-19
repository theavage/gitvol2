import torch
import glob
import PIL.Image as Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import os
 
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

