import torch
import torch.nn as nn
import argparse
import numpy as np
import os
from vqvae_training import VQVAE
from load_data import loadData, load_for_forward

class runVQVAE:
    def __init__(self, args, img_set):
        
        self.verbose = args.verbose
        self.mri_vqvae = VQVAE(args)
        self.img_set = img_set

        self.forward_run(args)

    @torch.no_grad()
    def forward_run(self, args):
        model = self.mri_vqvae

        if args.run_from_pre_trained:
            model.load_state_dict(torch.load(args.vqvae_load_ckpt, map_location=args.device), strict=False)

        train_data, val_data, test_data = load_for_forward(args)
        data = {'TRAINING_SET': train_data,
                'VALIDATION_SET': val_data,
                'TEST_SET': test_data}

        #mri_imgs = fake_dataset(1)
        steps_per_epoch = len(mri_imgs)
        all_index = {}
        all_imgs = {}
        i = 0
        print(f'-- RUNNING VQ-VAE FOR {self.img_set}--')
        actual_data = data[self.img_set]
        model.eval()
        for file in tqdm(actual_data):
            imgs = actual_data[file].type(torch.FloatTensor)
            imgs = imgs.to(args.device)
            decoded_images, index, _, = model(imgs)
            all_index[file] = index.cpu().numpy()
            if i % 4 and i < 20 == 0:
                all_imgs[file] = [decoded_images.cpu().detach().numpy(),
                                  imgs.cpu().detach().numpy()]
            i += 1

                   
        saveIndex(args, all_index, self.img_set)
        saveGeneratedImage(args, all_imgs, self.img_set)

        

