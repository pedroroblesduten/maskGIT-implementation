import torch
import torch.nn as nn
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

# Non-autoregressive generation
# Follows the original JAX implementation from Google Research: https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py
# But here in PyTorch!

class generateImages:
    def __init__(self, args, config):

        self.config = config
        self.transformer, self.vq_vae = self.getModels(args, config)
        self.codebook = self.vq_vae.codebook.embedding.weight.data
        self.loader = loadData(args)
                                        

        self.non_mask_confidence = float('inf')
        self.latent_dim = args.latent_dim
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

    def maskByConfidence(self, probs, mask_len):
        #The add term is just some random noise
        confidence = torch.log(probs) + args.temperature * torch.distributions.gumbel.Gumbel(0, 1).sample(probs.shape).to(logits.device)
        sorted_confidence, _ = torch.sort(confidence, dim=-1)
        cut_off = torch.take_along_dim(sorted_confidence, mask_len.to(torch.long), dim=-1)
        masking = (confidence < cut_off)
        return masking

    def createInputTokensNormal(self, batch_size, label):
        blank_tokens = torch.ones((batch_size, self.latent_dim), device=self.device)
        masked_tokens = self.mask_token * blank_tokens
        sos_tokens = torch.ones(inputs.shape[0], 1, dtype=torch.long, device=inputs.device) * self.sos_token

        if label is not None:
            label_tokens = label * torch.ones([batch_size, 1], device=label.device)
            label_tokens = label_tokens + self.mask_token
            masked_tokens = torch.cat((sos_tokens, label_tokens), dim=1)
            masked_tokens = torch.concat([label_tokens, masked_tokens], dim=-1)
        else:
            inputs = torch.cat((sos_tokens, masked_tokens), dim=1)
        
        return masked_tokens.to(torch.int32)

    def partialImageCreateInputTokens(self, idx, batch_size, label, start, end):
        #Create the input tokens passing just some part of the full sequence
        blank_tokens = torch.ones((batch_size, self.latent_dim), device=self.device)
        masked_tokens = self.mask_token * blank_tokens
        masked_tokens[start:end] = idx[start:end]
        sos_tokens = torch.ones(inputs.shape[0], 1, dtype=torch.long, device=inputs.device) * self.sos_token

        if label is not None:
            label_tokens = label * torch.ones([batch_size, 1], device=label.device)
            label_tokens = label_tokens + self.mask_token
            masked_tokens = torch.cat((sos_tokens, label_tokens), dim=1)
            masked_tokens = torch.concat([label_tokens, masked_tokens], dim=-1)
        else:
            inputs = torch.cat((sos_tokens, masked_tokens), dim=1)
        
        return masked_tokens.to(torch.int32)

    def getMaskRatio(self, iteration):
        ratio = 1. * (iteration + 1) / self.gen_iter
        return ratio


    def generateTokens(self, idx=None, label=None):
        #Getting inputs
        if idx is None:
            inputs = self.createInputTokensNormal(self.batch_size, label)
        else:
            inputs = self.partialImageCreateInputTokens(idx, self.batch_size, label, self.start, self.end)
        unknown_tokens_0 = torch.sum(inputs == self.mask_token_id, dim=-1)
        current_tokens = inputs
        for it in range(self.gen_iter):
            logits = self.transformer(inputs)
            pred_tokens = torch.distributions.categorical.Categorical(logits=logits).sample()
            unknown_tokens = (current_tokens == self.mask_token)
            sampled_tokens = torch.where(unknown_tokens, pred_tokens, current_tokens)

            r = self.getMaskRatio(it)
            mask_ratio = getMask(r, 'cosine')
            
            probs = F.softmax(logits, dim=-1)
            selected_probs = torch.squeeze(torch.take_along_dim(probs, torch.unsqueeze(sampled_tokens, -1), -1), -1)

            selected_probs = torch.where(unknown_tokens, selected_probs, self.non_mask_confidence)

            mask_len = torch.unsqueeze(torch.floor(unknown_tokens_0 * mask_ratio), 1)  
            mask_len = torch.maximum(torch.ones_like(mask_len), torch.minimum(torch.sum(unknown_tokens, dim=-1, keepdim=True)-1, mask_len))

            #Noise for randoness
            masking = self.maskByConfidence(selected_probs, temperature=4.5*(1.0 - r))

            current_tokens = torch.where(masking, self.mask_token, sampled_tokens)
        
        if label is None:
            generated_tokens = current_tokens[:, 1:]
        else:
            generated_tokens = current_tokens[:, 2:]

        return generated_tokens

    def decoderTokens(self, seq):
        img = self.vq_vae.decode(seq)
        return img

    def getReconstructionVQVAE(self, args):
        _, val_dataset = self.loader.getDataloader()
        print('-> GENERATING RESULTS FOR VQVAE <-')
        num_gen = 10
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
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    #VQ_VAE ARGS
    args.latent_dim = int(256)
    args.num_res_blocks = int(4)
    args.verbose = False
    args.num_codebook_vectors = int(256)
    args.beta = 0.25
    args.use_ema = True
    args.learning_rate = 2.25e-05
    args.beta1 = 0.5
    args.beta2 = 0.9
    
    #DATASET ARGS
    args.dataset = 'CIFAR10'
    args.imagenetPath = '/scratch2/pedroroblesduten/classical_datasets/imagenet'
    args.imagenetTxtPath = '/scratch2/pedroroblesduten/classical_datasets/imagenet/txt_files'
    args.cifar10Path = '/scratch2/pedroroblesduten/classical_datasets/cifar10'

    #TRAINING ARGS
    args.epochs = 200
    args.batch_size = 32
    args.device= 'cuda'
    args.patience = 10

    #PATH ARGS
    args.save_ckpt = '/scratch2/pedroroblesduten/MASKGIT/ckpt'
    args.save_losses = '/scratch2/pedroroblesduten/MASKGIT/losses'
    args.vqvae_load_ckpt = '/scratch2/pedroroblesduten/MASKGIT/ckpt/vqvae_bestVal_CIFAR10.pt'
    args.gpt_load_ckpt = None
    args.save_results_path = '/scratch2/pedroroblesduten/MASKGIT/results/'
    
    #TRANSFORMERS ARGS
    args.mask_token = 1025
    args.sos_token = 1024
    args.gen_iter = 8


    transformerConfig = MaskGITconfig(block_size = 257,
                                      vocab_size = 1026,
                                      n_layers = 10,
                                      n_heads = 8,
                                      embedding_dim = 768,
                                      dropout = 0.
                                      )



    generateImages(args, transformerConfig).getReconstructionVQVAE(args)
        
