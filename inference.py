import torch
import torch.nn as nn
import torch.nn.functional as F
from load_data import loadData
from encoder_decoder import Encoder, Decoder
from codebook import CodebookEMA
import argparse
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
from vqvae_training import VQVAE
import argparse
import math
from torchvision import utils as vutils
from maskgit_transformer import MaskGITconfig, MaskGITTransformer
from args_parameters import getArgs, getConfig
from utils import getMask

# Non-autoregressive generation
# Follows the original JAX implementation from Google Research: https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py
# But here in PyTorch!

class generateImages:
    def __init__(self, args, config):
        print('*preparing for inference*')

        self.config = config
        self.transformer, self.vq_vae = self.getModels(args, config)
        self.codebook = self.vq_vae.codebook.embedding.weight.data
        self.loader = loadData(args)
                                        

        self.non_mask_confidence = torch.Tensor([torch.inf]).to("cuda")
        self.block_size = 64
        self.device = args.device

        self.mask_token = args.mask_token
        self.sos_token = args.sos_token

        self.start = 40
        self.end = 80

        self.gen_iter = args.gen_iter
        self.batch_size = args.batch_size
        

        self.create_ckpt(args.save_results_path)
       
    def getModels(self, args, config):
        # LOADING GPT
        transf = MaskGITTransformer(config).to(args.device)
        if args.gpt_load_ckpt is not None:
            path  = args.gpt_load_ckpt.split('ckpt/')[-1]
            print(f' -> LOADING GPT MODEL: {path}')
            transf.load_state_dict(torch.load(args.gpt_load_ckpt, map_location=args.device), strict=False)
        else:
            print(f' -> LOADING TRANSFORMER MODEL: no checkpoint, intializing random')
                
        # LOADING VQ-VAE
        vq_vae = VQVAE(args).to(args.device)
        if args.vqvae_load_ckpt is not None:
            path = args.vqvae_load_ckpt.split('ckpt/')[-1]
            print(f' -> LOADING VQ-VAE MODEL: {path}')
            vq_vae.load_state_dict(torch.load(args.vqvae_load_ckpt, map_location=args.device), strict=False)

        return transf, vq_vae

    @staticmethod
    def create_ckpt(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def maskSequence(self, seq):
        z_indices = seq.view(x.shape[0], -1)

        sos_tokens = torch.ones(x.shape[0], 1, dtype=torch.long, device=z_indices.device) * self.sos_token

        r = math.floor(getMask(np.random.uniform(), 'cosine') * z_indices.shape[1])
        sample = torch.rand(z_indices.shape, device=z_indices.device).topk(r, dim=1).indices
        mask = torch.zeros(z_indices.shape, dtype=torch.bool, device=z_indices.device)
        mask.scatter_(dim=1, index=sample, value=True)

        masked_indices = self.mask_token * torch.ones_like(z_indices, device=z_indices.device)
        masked_indices = mask * z_indices + (~mask) * masked_indices

        masked_indices = torch.cat((sos_tokens, a_indices), dim=1)
        target = torch.cat((sos_tokens, z_indices), dim=1)

        return masked_indices, target, mask

    def maskByConfidence(self, probs, mask_len, temperature=1.0):
        #The add term is just some random noise
        confidence = torch.log(probs) + temperature * torch.distributions.gumbel.Gumbel(0, 1).sample(probs.shape).to(probs.device)
        sorted_confidence, _ = torch.sort(confidence, dim=-1)
        cut_off = torch.take_along_dim(sorted_confidence, mask_len.to(torch.long), dim=-1)
        masking = (confidence < cut_off)
        return masking

    def createInputTokensNormal(self, batch_size, label):
        blank_tokens = torch.ones((batch_size, self.block_size), device=label.device)
        masked_tokens = self.mask_token * blank_tokens
        sos_tokens = torch.full((batch_size, 1), self.sos_token, device=label.device)
        
        masked_tokens = torch.cat((sos_tokens, masked_tokens), dim=1)

        if label is not None:
            label_tokens = label + self.sos_token + 1
            label_tokens = label_tokens.unsqueeze(1)
            input_tokens = torch.cat((label_tokens, masked_tokens), dim=1)
        else:
            input_tokens = torch.cat((sos_tokens, masked_tokens), dim=1)
        
        return input_tokens.long()

    def partialImageCreateInputTokens(self, idx, batch_size, label, start, end):
        #Create the input tokens passing just some part of the full sequence
        
        idx = idx.view(batch_size, -1)
        blank_tokens = torch.ones((batch_size, self.block_size), device=label.device)
        masked_tokens = self.mask_token * blank_tokens
        masked_tokens[:, start:end] = idx[:, start:end]
        sos_tokens = torch.full((batch_size, 1), self.sos_token, device=label.device)

        if label is not None:
            label_tokens = label + self.sos_tokens + 1
            label_tokens = label.tokens.unsqueeze(1)
            input_tokens = torch.cat((label_tokens, masked_tokens), dim=1)
        else:
            input_tokens = torch.cat((sos_tokens, masked_tokens), dim=1)
        
        return input_tokens.long()

    def getMaskRatio(self, iteration):
        ratio = 1. * (iteration + 1) / self.gen_iter
        return ratio


    def generateTokens(self, label=None, idx=None, start=20, end=50):
        #GETTING INPUTS
        if idx is None:
            inputs = self.createInputTokensNormal(self.batch_size, label)
        else:
            inputs = self.partialImageCreateInputTokens(idx, self.batch_size, label, start, end)
        unknown_tokens_0 = torch.sum(inputs == self.mask_token, dim=-1)
        current_tokens = inputs
        
        #RUN PRE-TRAINED TRANSFORMER
        for it in range(self.gen_iter):
            self.transformer.eval()
            with torch.no_grad():
                logits = self.transformer(inputs)

            pred_tokens = torch.distributions.categorical.Categorical(logits=logits).sample()
            unknown_tokens = (current_tokens == self.mask_token)
            sampled_tokens = torch.where(unknown_tokens, pred_tokens, current_tokens)

            r = self.getMaskRatio(it)
            mask_ratio = getMask(r, 'cosine')
            
            probs = F.softmax(logits, dim=-1)
            selected_probs = torch.squeeze(torch.take_along_dim(probs, torch.unsqueeze(sampled_tokens, -1), -1), -1).float()

            selected_probs = torch.where(unknown_tokens, selected_probs, self.non_mask_confidence)

            mask_len = torch.unsqueeze(torch.floor(unknown_tokens_0 * mask_ratio), 1)  
            mask_len = torch.maximum(torch.ones_like(mask_len), torch.minimum(torch.sum(unknown_tokens, dim=-1, keepdim=True)-1, mask_len))

            #Noise for randoness
            masking = self.maskByConfidence(selected_probs, mask_len, 4.5*(1.0 - r))

            current_tokens = torch.where(masking, self.mask_token, sampled_tokens)
        
        if label is None:
            generated_tokens = current_tokens[:, 1:]
        else:
            generated_tokens = current_tokens[:, 2:]

        return generated_tokens

    
    def latentFromSequence(self, args, seq):
        sequence = torch.flatten(seq)[:, None]
        #encoding_indices = sequence.type(torch.int64).to(args.device)
        encodings = torch.zeros(sequence.shape[0], args.num_codebook_vectors, device=args.device)
        encodings.scatter_(1, sequence, 1)
        z_q = torch.matmul(encodings, self.codebook).view((seq.shape[0], 8, 8, 256))
        z_q = z_q.permute(0, 3, 1, 2).to(args.device)

        return z_q

        
    @torch.no_grad()
    def fullGenerationProcess(self, args, imgs, label, i=0):
        self.vq_vae.eval()
        with torch.no_grad():
            decoded, indices = self.vq_vae.encode(imgs)
            generated_tokens = self.generateTokens(label)
            latent_space = self.latentFromSequence(args, generated_tokens)
            generated_images = self.vq_vae.decode(latent_space)

        real_fake_images = torch.cat((imgs[:4], generated_images.add(1).mul(0.5)[:4]))
        vutils.save_image(real_fake_images, os.path.join(args.save_results_path, f'generated_imgs_{i}.jpg'), nrow=4)
        i = i +1
        return i

    def generation(self, args):
        train_dataset, val_dataset = self.loader.getDataloader()
        print(f' -> GENERATING IMAGES FOR {args.dataset} <- ')
        j = 0
        with torch.no_grad():
            for imgs, label in tqdm(val_dataset):
                imgs, label = imgs.to(args.device), label.to(args.device)
                j = self.fullGenerationProcess(args, imgs, label, j)
                if j == 5:
                    break         


    def getReconstructionVQVAE(self, args):
        _, val_dataset = self.loader.getDataloader()
        print('-> GENERATING RESULTS FOR VQVAE <-')
        num_gen = 20
        i = 0
        self.vq_vae.eval()
        for imgs, _ in val_dataset:
            imgs = imgs.to(args.device)
            decoded_images, indices, loss = self.vq_vae(imgs)
            real_fake_images = torch.cat((imgs[:4], decoded_images.add(1).mul(0.5)[:4]))
            vutils.save_image(real_fake_images, os.path.join(args.save_results_path, f'results_vqvae_{i}.jpg'), nrow=4)
            i+=1
            if i == num_gen:
                break


if __name__ == '__main__': 
    
    arg = getArgs()
    conf = getConfig()
    
    #generateImages(arg, conf).getReconstructionVQVAE(arg)

    #GENERATING IMAGES
    generateImages(arg, conf).generation(arg)


    
        
