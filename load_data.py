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


def loadData(args):
    all_imgs = os.listdir(args.dataset_path)
    #all_imgs = random.shuffle(all_imgs)
    data_len = len(all_imgs)

    s_train = data_len*0.8
    s_val = data_len*0.1 + s_train
    s_test = data_len*0.1 + s_val

    train_set = all_imgs[:s_train]
    val_set = all_imgs[s_train:s_val]a
    test_set = all_imgs[s_val:s_test]
    

    train_data = ImagePaths(args.dataset_path, train_set size=256)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    train_data = ImagePaths(args.dataset_path, val_set size=256)
    val_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    train_data = ImagePaths(args.dataset_path, test_set size=256)
    test_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    return train_loader, val_loader, test_loader
