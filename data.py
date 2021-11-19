import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')


class CustomImageFolder(ImageFolder):

    def __init__(self, root, transform = None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img



class tensorDataset(Dataset):

    def __init__(self, data):
        # data : data tensor
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]


def dsprites(path, batch_size):
    # path : numpy array path

    data = np.load(path, encoding = 'bytes', allow_pickle = True) # pxiels are already normalized by /255
    print('dsprites dataset shape : ' + str(data['imgs'].shape)) # 737280 * 64 * 64
    data = torch.from_numpy(data['imgs']).unsqueeze(1).float() # num_images * 1 * 64 * 64

    train_dataset = tensorDataset(data)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True)
    #batch = iter(train_loader).next()
    # print(batch.shape) # batch_size * 1 * 64 * 64
    return train_loader



def celebA(root, batch_size, img_h=64, img_w=64):
    # path : path to root images folder
    # 202596 images

    transform = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor()])
    train_dataset = CustomImageFolder(root, transform)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    
    print("chose celebA dataset")
    print("total number of images : " + str(len(train_loader.dataset)))

    return train_loader


def chairs(root, batch_size, img_h, img_w):
    # path : path to root folder of dataset

    transform = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor()])
    train_dataset = CustomImageFolder(root, transform)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

    print("chose 3dcharis dataset")
    print("total number of images : " + str(len(train_loader.dataset)))

    return train_loader


def data(dataset_name, batch_size, img_h, img_w) :

    if dataset_name == 'dsprites':
        return dsprites('data/dsprites/dsprites.npz', batch_size = batch_size)
    if dataset_name == 'celebA':
        return celebA('data/celebA', batch_size = batch_size, img_h = img_h, img_w = img_w)
    if dataset_name == '3dchairs':
        return chairs('data/rendered_chairs', batch_size = batch_size, img_h = img_h, img_w = img_w)



