import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from load_data import loadIndex
from run_vqvae import runVQVAE
from tqdm import tqdm
from my_minGPT import GPT, GPTconfig
from maskgit_transformer import MaskGITTransformer, MaskGITconfig
import argparse
import math

# Non-autoregressive generation
# Follows the original JAX implementation from Google Research: https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py
# But here in PyTorch!

class generateImages:
    def __init__(self, args, config):

        self.config = config
        self.transformer, self.vq_vae = self.getModels(args, config)
        self.codebook = self.vq_vae.codebook.embedding.weight.data

        self.non_mask_confidence = float('inf')
        self.latent_dim = args.latent_dim
        self.device = args.device
        self.mask_token = args.mask_token
        self.sos_token = args.sos_token
        self.gen_iter = args.gen_iter
        self.batch_size = args.batch_size


        self.create_ckpt(args.gpt_save_ckpt)
        self.train(args, config, run_vqvae)
        

    def getModels(self, args, config):
        # LOADING GPT
        transf = GPT(config).to(args.device)
        if args.gpt_load_ckpt is not None:
            path  = args.gpt_load_ckpt.split('ckpt/')[-1]
            print(f' -> LOADING GPT MODEL: {path}')
            transf.load_state_dict(torch.load(args.gpt_load_ckpt, map_location=args.device), strict=False)
                
        # LOADING VQ-VAE
        vq_vae = MRI_VQVAE(args).to(args.device)
        if args.vqvae_load_ckpt is not None:
            path = args.vqvae_load_ckpt.split('ckpt/')[-1]
            print(f' -> LOADING VQ-VAE MODEL: {path}')
            vq_vae.load_state_dict(torch.load(args.vqvae_load_ckpt, map_location=args.device), strict=False)

        return transf, vq_vae

    @staticmethod
    def create_ckpt(ckpt_path):
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

    def maskSequence(seq):
        z_indices = seq.view(x.shape[0], -1)

        sos_tokens = torch.ones(x.shape[0], 1, dtype=torch.long, device=z_indices.device) * 1024

        r = math.floor(getMask(np.random.uniform(), 'cosine') * z_indices.shape[1])
        sample = torch.rand(z_indices.shape, device=z_indices.device).topk(r, dim=1).indices
        mask = torch.zeros(z_indices.shape, dtype=torch.bool, device=z_indices.device)
        mask.scatter_(dim=1, index=sample, value=True)

        masked_indices = 1025 * torch.ones_like(z_indices, device=z_indices.device)
        masked_indices = mask * z_indices + (~mask) * masked_indices

        masked_indices = torch.cat((sos_tokens, a_indices), dim=1)
        target = torch.cat((sos_tokens, z_indices), dim=1)

        return masked_indices, target, mask

    def maskByConfidence(probs):
        #probs = F.softmax(logits, dim=-1)
        confidence = torch.log(probs) + args.temperature * torch.distributions.gumbel.Gumbel(0, 1).sample(probs.shape).to(logits.device)
        sorted_confidence, _ = torch.sort(confidence, dim=-1)
        cut_off = torch.take_along_dim(sorted_confidence, mask_len.to(torch.long), dim=-1)
        masking = (confidence < cut_off)
        return masking

    def createInputTokensNormal(self, batch_size, label):
        blank_tokens = torch.ones((num, self.latent_dim), device=self.device)
        masked_tokens = self.mask_token * blank_tokens
        sos_tokens = torch.ones(inputs.shape[0], 1, dtype=torch.long, device=inputs.device) * self.sos_token

        if label is not None:
            label_tokens = label * torch.ones([batch_size, 1])
            label_tokens = label_tokens + self.latent_dim
            input_tokens = torch.concat([label_tokens, masked_tokens], dim=-1)
            inputs = torch.cat((sos_tokens, input_tokens), dim=1)
        else:
            inputs = torch.cat((sos_tokens, masked_tokens), dim=1)
        
        return masked_tokens.to(torch.int32)

    def getMaskRatio(iteration):
        return ratio = 1. * (iteration + 1) / self.gen_iter
    


    def generateTokens(self, idx=None, label=None):
        unknown_tokens_0 = torch.sum(inputs == self.mask_token, dim=-1)

        #Getting inputs
        if idx is None:
            inputs = createInputTokensNormal(self.batch_size, label)
        else:
            inputs = torch.hstack((idx, torch.zeros((inputs.shape[0], N - inputs.shape[1]), device="cuda", dtype=torch.int).fill_(self.mask_token_id)))
            sos_tokens = torch.ones(inputs.shape[0], 1, dtype=torch.long, device=inputs.device) * self.sos_token
            inputs = torch.cat((sos_tokens, masked_tokens), dim=1)

        current_tokens = inputs
        for it in range(self.gen_iter):
            logits = self.transformer(inputs)
            pred_tokens = torch.distributions.categorical.Categorical(logits=logits).sample()
            unknown_tokens = (current_tokens == self.mask_token)
            sampled_tokens = torch.where(unknown_tokens, pred_tokens, current_tokens)

            r = getMaskRatio(it)
            mask_ratio = getMask(mask_schedule_ratio, 'cosine')
            
            probs = F.softmax(logits, dim=-1)
            selected_probs = torch.squeeze(torch.take_along_dim(probs, torch.unsqueeze(sampled_tokens, -1), -1), -1)

            selected_probs = torch.where(unknown_map, selected_probs, self.non_mask_confidence)

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
        
