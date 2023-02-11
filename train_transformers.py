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

#Training for GPT follows: https://github.com/karpathy/nanoGPT/blob/master/train.py

class trainTransformers:
    def __init__(self, args, config):

        self.config = config
        self.transformer, self.vq_vae = self.getModels(args, config)
        self.codebook = self.vq_vae.codebook.embedding.weight.data

        self.sos_token = args.sos_token
        self.mask_token = args.mask_token

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

    def train(self, args, config, run_vqvae):

        train_images = load_data
        val_images = load_data


        model = MaskGITTransformer(self.config)

        model.to(args.device)

        opt_dict = dict(
            learning_rate = 6e-4,
            weight_decay = 1e-2,
            betas = (0.9,0.95)
        )

        optimizer = model.configure_optimizers(opt_dict['weight_decay'],
                                               opt_dict['learning_rate'],
                                               opt_dict['betas'])
        def get_lr(iter):
            # 1) linear warmup for warmup_iters steps
            if iter < warmup_iters:
                return learning_rate * iter / warmup_iters
            # 2) if iter > lr_decay_iters, return min learning rate
            if iter > lr_decay_iters:
                return min_lr
            # 3) in between, use cosine decay down to min learning rate
            decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
            return min_lr + coeff * (learning_rate - min_lr)

        def maskSequence(seq, label=None):
            #TODO: available training for no label data in the same model
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

            if label is not None:
                label_tokens = label * torch.ones([batch_size, 1])
                label_tokens = label_tokens + self.mask_token
                input_tokens = torch.concat([label_tokens, masked_indices], dim=-1)
                target_tokens = torch.concat([label_tokens, target], dim=-1)

            return input_tokens.long(), target_tokens.long()


        # -- TRAINING LOOP PARAMETERS -- 

        decay_lr = False
        max_iters = len(X_train_index)*args.gpt_epochs
        warmup_iters = len(X_train_index)*5
        lr_decay_iters = len(X_train_index)*args.gpt_epochsa
        learning_rate = 6e-4 # max learning rate
        min_lr = 6e-5

        eval_interval = 10
        patience = args.patience
        patience_counter = 0
        iter_num = 0
        best_val_loss = 1e9

        # -----------------------------


        ######################
        #      TRAINING      #
        #        LOOP        #
        ######################

        print(f'--- STARTING MASKGIT TRANSFORMER TRAINING ---')
        iter_num = 0
        all_train_loss = []
        all_val_loss = []

        for epoch in range(args.gpt_epochs):
            epoch_val_losses = []
            epoch_train_losses = []

            model.train()
            for imgs in tqdm(train_images):

                if decay_lr:
                    lr = get_lr(iter_num)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = learning_rate
                   
                #RUNNING VQ_VAE 
                imgs = imgs.to(args.device)
                _, indices = self.vq_vae.encode(x)
                
                #MASKING SEQUENCE FOR TRANSFORMER TRAINING
                masked_indices, target = maskSequence(indices)

                #RUNNING TRANSFORMER
                logits = self.transformer(masked_indices)

                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))            

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                iter_num += 1
                epoch_train_losses.append(loss.detach().cpu().numpy())
                    
                    
            model.eval()
            for val_batch in X_val_index:                
                imgs = imgs.to(args.device)
                _, indices = self.vq_vae.encode(x)                
                masked_indices, target = maskSequence(indices)
                logits = self.transformer(masked_indices)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))          
                epoch_val_losses.append(loss.detach().cpu().numpy())

            train_loss, val_loss = np.mean(epoch_train_losses), np.mean(epoch_val_losses)
 
            #Early Stop
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(args.gpt_save_ckpt, 'scale_GPT_bestVAL.pt'))

            else:
                patience_counter += 1

            # if patience_counter > patience:
            #    break
            
            all_train_loss.append(train_loss)
            all_val_loss.append(val_loss)

            if epoch % 10 == 0:
                np.save(os.path.join(args.save_loss, 'scale_GPT_train_loss.npy'), all_train_loss)
                np.save(os.path.join(args.save_loss, 'scale_GPT_val_loss.npy'), all_val_loss)
                torch.save(model.state_dict(), os.path.join(args.gpt_save_ckpt, f'scale_GPT_lastEpoch_{epoch}.pt'))
       
        print('--- FINISHED MASKGIT TRANSFORMER TRAINING ---')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GPT_TRAINING")

    # VQVAE ARGS
    parser.add_argument('--latent-dim', type=int, default=256)
    parser.add_argument('--num-codebook-vectors', type=int, default=1024)
    parser.add_argument('--beta', type=float, default=0.25)
    parser.add_argument('--image-channels', type=int, default=1)
    parser.add_argument('--flat_ordering', type=str, default=None)
    parser.add_argument('--use_ema', type=str, default=True)


    # GENERAL TRAINING ARGS
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=2.25e-05)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--verbose', type=str, default=False)
    parser.add_argument('--img_sets', type=str, default=['train_set', 'validation_set', 'test_set'])

    #LOAD AND SAVE ARGS:
    parser.add_argument('--save_mode', type=str, default='training')
    parser.add_argument('--run_from_pre_trained', type=bool, default=True)

    # PATH ARGS
    parser.add_argument('--save_mri_path', type=str, default='./generated_images')
    parser.add_argument('--train_csv_path', type=str, default='/scratch2/pedroroblesduten/BRAIN_COVID/brain_anomaly_detection/vq_vae_3d/norep_treino_3_classes_onehot.csv')
    parser.add_argument('--validation_csv_path', type=str, default='/scratch2/pedroroblesduten/BRAIN_COVID/brain_anomaly_detection/vq_vae_3d/norep_validacao_3_classes_onehot.csv')
    parser.add_argument('--test_csv_path', type=str, default='/scratch2/pedroroblesduten/BRAIN_COVID/brain_anomaly_detection/vq_vae_3d/norep_teste_3_classes_onehot.csv')
    parser.add_argument('--dataset_path', type=str, default='/scratch2/turirezende/BRAIN_COVID/data/ADNI/images')
    parser.add_argument('--gpt_save_ckpt', type=str, default="/scratch2/pedroroblesduten/BRAIN_COVID/brain_anomaly_detection/vq_vae_3d/ckpt/")
    parser.add_argument('--gpt_load_ckpt', type=str, default='/scratch2/pedroroblesduten/BRAIN_COVID/brain_anomaly/detection/vq_vae_3d/ckpt/')
    parser.add_argument('--index_path', type=str, default='/scratch2/pedroroblesduten/BRAIN_COVID/brain_anomaly_detection/vq_vae_3d/save_index')
    parser.add_argument('--vqvae_load_ckpt', type=str, default='/scratch2/pedroroblesduten/BRAIN_COVID/brain_anomaly_detection/vq_vae_3d/ckpt/mri_vqvae_bestVal.pt')
    parser.add_argument('--save_loss', type=str, default='/scratch2/pedroroblesduten/BRAIN_COVID/brain_anomaly_detection/vq_vae_3d/losses/')
    # GPT ARGS
    parser.add_argument('--gpt_batch_size', type=str, default=2)
    parser.add_argument('--gpt_epochs', type=str, default=500)
    parser.add_argument('--patience', type=int, default=10)


    args = parser.parse_args()
    args.gpt_load_ckpt = None
    # args.verbose = True
    
    # TRANSFORMER ARCHITECURE CONFIG PARAMETERS
    transformerConfig = MaskGITconfig(block_size = 257,
                                      vocab_size = 1026,
                                      n_layers = 10,
                                      n_heads = 8,
                                      embedding_dim = 768,
                                      dropout = 0.
                                      )


    print(f'GPT CONFIGURATIONS: {transformerConfig}')
    train_GPT = trainTransformers(args, transformerConfig)
