import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transf = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        path, label = self.data_list[index]
        image = Image.open(path)
        if self.transf:
            image = self.transf(image)
        return (image.unsqueeze(0), torch.tensor(label, dtype=torch.long))

class loadData:
    def __init__(self, args):
  
        
        self.batch_size = args.batch_size

        self.dataset = args.dataset
        if self.dataset == 'ImageNet':
            self.dataPath = args.imagenetPath
            self.txtPath = args.imagenetTxtPath
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.create_paths(self.dataPath)


        
    @staticmethod
    def create_paths(path):
        if not os.path.exists(path):
            os.makedirs(path)


    def readTxt(self, txt_file, i_set):
        data = []
        with open(txt_file, 'r') as f:
            for line in f:
                image_file, label = line.strip().split()
                image_path = os.path.join(self.dataPath, i_set, image_file+'.JPEG').split(' ')[0]
                with Image.open(image_path) as image:
                    data.append((image, int(label)))
        return data

    def getDataloader(self):
        print(' -> Loading data... ')

        trainPath = os.path.join(self.dataPath, 'train')
        
        val_txt = os.path.join(self.txtPath, 'val.txt')
        test_txt = os.path.join(self.txtPath, 'test.txt')

        train_dataloader=DataLoader(
                torchvision.datasets.ImageFolder(trainPath,transform=self.transform),
                batch_size=self.batch_size, shuffle=True
            )

        val_dataset = ImageDataset(self.readTxt(val_txt, 'val'), self.transform)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

        #test_dataset = ImageDataset(self.readTxt(test_txt, 'test'), transform=transform)
        #test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

        return train_dataloader, val_dataloader #, test_dataloader






