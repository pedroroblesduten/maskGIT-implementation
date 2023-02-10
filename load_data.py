import random
import os
import albumentations
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
#Based on: https://github.com/dome272/MaskGIT-pytorch/blob/main/utils.py
class ImagePaths(Dataset):
    def __init__(self, path, img_set, size=None):
        self.size = size

        self.images = [os.path.join(path, file) for file in img_set]
        self._length = len(self.images)

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, i):
        example = self.preprocess_image(self.images[i])
        return example


def loadData(args, batch_size=None):
    all_imgs = os.listdir(args.dataset_path)
    #all_imgs = random.shuffle(all_imgs)
    data_len = len(all_imgs)
    if batch_size is not None:
        batchsize = args.batch_size
    else:
        batchsize = 1

    s_train = data_len*0.8
    s_val = data_len*0.1 + s_train
    s_test = data_len*0.1 + s_val

    train_set = all_imgs[:s_train]
    val_set = all_imgs[s_train:s_val]
    test_set = all_imgs[s_val:s_test]
    

    train_data = ImagePaths(args.dataset_path, train_set, size=256)
    train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True)

    train_data = ImagePaths(args.dataset_path, val_set, size=256)
    val_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True)

    train_data = ImagePaths(args.dataset_path, test_set, size=256)
    test_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True)

    return train_loader, val_loader, test_loader

def load_for_forward(args):
    train_loader, val_loader, test_loader = loadDat(args, 1)
    train, val, test = {}, {}, {}
    for i, img in enumerate(train_loader):
        train[f'train_image_{i}'] = img
    for i, img in enumerate(val_loader):
        val[f'val_image_{i}'] = img
    for i, img in enumerate(test_loader):
        test[f'test_image_{i}'] = img
    
    return train, val, test


def saveIndex(args, idx_dict, img_set):
    save_path = args.save_index_path
    full_path = os.path.join(save_path, img_set)
    for img_name in idx_dict:
        idx = idx_dict[img_name]
        np.save(os.path.join(full_path, img_name), idx+'.npy')

def loadIndexForward(args, img_set):
    save_path = args.save_index_path
    full_path = os.path.join(save_path, img_set)
    file_name = os.listdir(full_path)
    dict_idx = {}, {}, {}
    for img_name in file_name:
        path = os.path.join(full_path, img_name)
        dict_idx[img_name.split('.npy')[0]] = np.load(path)
    return dict_idx

def loadIndex(args, img_set):
    path = os.path.join(self.save_indx, index_set)
        
    arrays_files = os.listdir(path)
    list_of_arrays = []
    for file in arrays_files:
        array = np.load(os.path.join(path, file))
            
        list_of_arrays.append(torch.tensor(array))

    tensor_of_tensors = torch.stack(list_of_arrays)

    dataloader = DataLoader(tensor_of_tensors, batch_size=batch_size, shuffle=True)
    return dataloader

def saveGeneratedImage(args, img_dict, img_set):
    for file in img_dict:
        original, reconstruction = img_dict[file][0], img_dict[file][1]
        original = original[0, :, :, :]
        reconstruction = reconstruction[0, :, :, :]

        fig, axarr = plt.subplots(1, 2)
        axarr[0].imshow(original.transpose(1, 2, 0))
        axarr[1].imshow(reconstruction.cpu().detach().numpy()[0].transpose(1, 2, 0))
        fig.savefig(args.save_generated)
        plt.close()

    







