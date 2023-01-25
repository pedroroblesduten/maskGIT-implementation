import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from load_data import loadIndex
from utils import fake_dataset
from run_vqvae import runVQVAE
from tqdm import tqdm
from my_minGPT import GPT, GPTconfig
from my_minGPT import MaskGITTransformers, GPTconfig
import argparse
import math

#Training for GPT follows: https://github.com/karpathy/nanoGPT/blob/master/train.py

class trainGPT:
    def __init__(self, args, config, run_vqvae, training):

        self.load_index = LoadSaveIndex(args)
        self.config = config
        self.flat_ordering = args.flat_ordering
        self.create_ckpt(args.gpt_save_ckpt)
        self.train(args, config, run_vqvae)

    @staticmethod
    def create_ckpt(ckpt_path):
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

    def train(self, args, config, run_vqvae):
        if run_vqvae:
            for img_set in self.img_sets:
                run_vq = runVQVAE(args, img_set)

        torch.cuda.empty_cache()

        X_train_index = loadIndex(args.gpt_batch_size, self.img_sets[0])
        
        X_val_index = loadIndex(args.gpt_batch_size, self.img_sets[1])

        model = MaskGITTransformers(self.config)

        if args.gpt_load_ckpt is not None:
            model.load_state_dict(args.gpt_load_ckpt)

        model.to(args.device)

        opt_dict = dict(
            learning_rate = 6e-4,
            max_iters = 600000, 
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

        def get_inputs(z_indices, pkeep):

            if pkeep < 1.0:
                mask = torch.bernoulli(pkeep*torch.ones(z_indices.shape,
                                                         device=z_indices.device))
                mask = mask.round().to(dtype=torch.int64)
            
                r_indices = torch.randint_like(z_indices, self.config.vocab_size)
                X_indices = mask*z_indices+(1-mask)*r_indices
            else:
                X_indices = z_indices
                
            pad_X = (1, 0, 0, 0)
            X_indices = F.pad(a_indices, pad_X, value=1024)
            pad_Y = (0, 1, 0, 0)
            Y_indices = F.pad(z_indices, pad_Y, value=1024)

            return X_indices, Y_indices


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


        # --- TRAINING ---

        print(f'--- STARTING GPT TRAINING FOR {self.classes[1]} ---')
        iter_num = 0
        all_train_loss = []
        all_val_loss = []
        for epoch in range(args.gpt_epochs):
            epoch_val_losses = []
            epoch_train_losses = []

            model.train()
            for batch in tqdm(X_train_index):

                if decay_lr:
                    lr = get_lr(iter_num)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = learning_rate
                    
                X, Y = get_inputs(batch, args.pkeep)
                X, Y = X.to(args.device), Y.to(args.device)

                optimizer.zero_grad(set_to_none=True)
                logits, loss = model(X, Y)
                loss.backward()
                optimizer.step()
                iter_num += 1
                epoch_train_losses.append(loss.detach().cpu().numpy())
                    
                    
            model.eval()
            for val_batch in X_val_index:
                X, Y = get_inputs(val_batch, args.pkeep)
                X, Y = X.to(args.device), Y.to(args.device)
                logits, loss = model(X, Y)
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
       
        print('--- FINISHED GPT TRAINING ---')

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from load_mri import LoadMRI, SaveMRI, LoadSaveIndex
from utils import fake_dataset
from run_vqvae_from_ckpt import MriRunVQVAE
from tqdm import tqdm
from my_minGPT import GPT, GPTconfig
import argparse
import math

#Training for GPT follows: https://github.com/karpathy/nanoGPT/blob/master/train.py

class trainMaskGITTransformers:
    def __init__(self, args, config, run_vqvae, training):

        self.load_index = LoadSaveIndex(args)
        self.config = config
        self.verbose = args.verbose
        self.classes = args.classes
        self.img_sets = args.img_sets
        self.train_on = training
        self.flat_ordering = args.flat_ordering
        self.create_ckpt(args.gpt_save_ckpt)
        self.train(args, config, run_vqvae)

    @staticmethod
    def create_ckpt(ckpt_path):
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

    def train(self, args, config, run_vqvae):
        if run_vqvae:
            for classe in self.classes:
                for img_set in self.img_sets:

                    run_vq = MriRunVQVAE(args, img_set, classe, self.flat_ordering)

        torch.cuda.empty_cache()

        X_train_index = self.load_index.loadIndex(args.gpt_batch_size,
                                                  self.classes[1],
                                                  self.img_sets[0])
        
        X_val_index = self.load_index.loadIndex(args.gpt_batch_size,
                                                self.classes[1],
                                                self.img_sets[1])

        model = MaskGITTransformers(self.config)

        if args.gpt_load_ckpt is not None:
            model.load_state_dict(args.gpt_load_ckpt)

        model.to(args.device)

        opt_dict = dict(
            learning_rate = 6e-4,
            max_iters = 600000, 
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

        def get_inputs(z_indices, pkeep):

            if pkeep < 1.0:
                mask = torch.bernoulli(pkeep*torch.ones(z_indices.shape,
                                                         device=z_indices.device))
                mask = mask.round().to(dtype=torch.int64)
            
                r_indices = torch.randint_like(z_indices, self.config.vocab_size)
                a_indices = mask*z_indices+(1-mask)*r_indices
            else:
                a_indices = z_indices
                
            pad_X = (1, 0, 0, 0)
            a_indices = F.pad(a_indices, pad_X, value=1024)
            pad_Y = (0, 1, 0, 0)
            z_indices = F.pad(z_indices, pad_Y, value=1024)

            return a_indices, z_indices


        # -- TRAINING LOOP PARAMETERS -- 

        decay_lr = False
        max_iters = len(X_train_index)*args.gpt_epochs
        warmup_iters = len(X_train_index)*5
        lr_decay_iters = len(X_train_index)*args.gpt_epochs
        learning_rate = 6e-4 # max learning rate
        min_lr = 6e-5

        eval_interval = 10
        patience = args.patience
        patience_counter = 0
        iter_num = 0
        best_val_loss = 1e9


        # --- TRAINING ---

        print(f'--- STARTING GPT TRAINING FOR {self.classes[1]} ---')
        iter_num = 0
        all_train_loss = []
        all_val_loss = []
        for epoch in range(args.gpt_epochs):
            epoch_val_losses = []
            epoch_train_losses = []

            model.train()
            for batch in tqdm(X_train_index):

                if decay_lr:
                    lr = get_lr(iter_num)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = learning_rate
                    
                X, Y = get_inputs(batch, args.pkeep)
                X, Y = X.to(args.device), Y.to(args.device)

                optimizer.zero_grad(set_to_none=True)
                logits, loss = model(X, Y)
                loss.backward()
                optimizer.step()
                iter_num += 1
                epoch_train_losses.append(loss.detach().cpu().numpy())
                    
                    
            model.eval()
            for val_batch in X_val_index:
                X, Y = get_inputs(val_batch, args.pkeep)
                X, Y = X.to(args.device), Y.to(args.device)
                logits, loss = model(X, Y)
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
       
        print('--- FINISHED GPT TRAINING ---')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GPT_TRAINING")

    # VQVAE ARGS
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=1, help='Number of channels of images (default: 1)')
    parser.add_argument('--flat_ordering', type=str, default=None, help='How to flatten the latent space')
    parser.add_argument('--use_ema', type=str, default=True, help='If will use EMA for codebook update')


    # GENERAL TRAINING ARGS
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=8, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--verbose', type=str, default=False, help='Verbose to control prints in the foward pass')
    parser.add_argument('--classes', type=list, default=['AD', 'CN', 'MCI'], help='Classes in the adni_dataset')
    parser.add_argument('--img_sets', type=str, default=['train_set', 'validation_set', 'test_set'], help='Machine learning traditional sets of data')

    #LOAD AND SAVE ARGS:
    parser.add_argument('--save_mode', type=str, default='training', help='How we want to save MRI images')
    parser.add_argument('--run_from_pre_trained', type=bool, default=True, help='If true, run vq_vae from a pre-trained checkpoint')

    # PATH ARGS
    parser.add_argument('--save_mri_path', type=str, default='./generated_images', help='Path for save autoencoder outputs')
    parser.add_argument('--train_csv_path', type=str, default='/scratch2/pedroroblesduten/BRAIN_COVID/brain_anomaly_detection/vq_vae_3d/norep_treino_3_classes_onehot.csv')
    parser.add_argument('--validation_csv_path', type=str, default='/scratch2/pedroroblesduten/BRAIN_COVID/brain_anomaly_detection/vq_vae_3d/norep_validacao_3_classes_onehot.csv')
    parser.add_argument('--test_csv_path', type=str, default='/scratch2/pedroroblesduten/BRAIN_COVID/brain_anomaly_detection/vq_vae_3d/norep_teste_3_classes_onehot.csv')
    parser.add_argument('--dataset_path', type=str, default='/scratch2/turirezende/BRAIN_COVID/data/ADNI/images')
    parser.add_argument('--gpt_save_ckpt', type=str, default="/scratch2/pedroroblesduten/BRAIN_COVID/brain_anomaly_detection/vq_vae_3d/ckpt/", help='Path to save gpt weights checkpoints')
    parser.add_argument('--gpt_load_ckpt', type=str, default='/scratch2/pedroroblesduten/BRAIN_COVID/brain_anomaly/detection/vq_vae_3d/ckpt/', help='Path to load gpt weights checkpoints')
    parser.add_argument('--index_path', type=str, default='/scratch2/pedroroblesduten/BRAIN_COVID/brain_anomaly_detection/vq_vae_3d/save_index', help='Path to save and load indices from encoder')
    parser.add_argument('--vqvae_load_ckpt', type=str, default='/scratch2/pedroroblesduten/BRAIN_COVID/brain_anomaly_detection/vq_vae_3d/ckpt/mri_vqvae_bestVal.pt', help='Path to load thw weights of a pre-trained vq-vae')
    parser.add_argument('--save_loss', type=str, default='/scratch2/pedroroblesduten/BRAIN_COVID/brain_anomaly_detection/vq_vae_3d/losses/')
    # GPT ARGS
    parser.add_argument('--gpt_batch_size', type=str, default=2, help='Batch size for gpt training')
    parser.add_argument('--gpt_epochs', type=str, default=500, help='Number of epochs for gpt training')
    parser.add_argument('--patience', type=int, default=10, help='Number of iterations before early stop')
    parser.add_argument('--pkeep', type=float, default=1.0, help='Probability of a token be changed before transformer')
    parser.add_argument('--tumor_path', type=str, default='/scratch2/pedroroblesduten/BRAIN_COVID/brain_anomaly_detection/vq_vae_3d/fake_tumor')



    args = parser.parse_args()
    args.gpt_load_ckpt = None
    # args.verbose = True
    
    # GPT ARCHITECURE CONFIG PARAMETERS
    gptconf = GPTconfig(block_size = 1574, # how far back does the model look? i.e. context size
                        vocab_size = 1025,
                        n_layers = 10,
                        n_heads = 8,
                        embedding_dim = 768, # size of the model
                        dropout = 0.# for determinism
                        )


    print(f'GPT CONFIGURATIONS: {gptconf}')
    train_GPT = trainTransformers(args, gptconf, run_vqvae=True, training=True)
