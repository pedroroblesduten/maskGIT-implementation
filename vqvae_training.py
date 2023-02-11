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

class VQVAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.verbose = args.verbose
        self.use_ema = args.use_ema

        self.encoder = Encoder(args).to(device=args.device)
        self.decoder = Decoder(args).to(device=args.device)
        # if self.use_ema:
        self.codebook = CodebookEMA(args).to(device=args.device)
        #else:
        #    self.codebook = Codebook3D(args, verbose=self.verbose).to(device=args.device)
        self.quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device)
        self.post_quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device)

    def forward(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        if self.verbose:
            print(f'Shape before codebook: {quant_conv_encoded_images.shape}')
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_images)
        if self.verbose:
            print(f'Shape before decoder: {codebook_mapping.shape}')
        post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        decoded_images = self.decoder(post_quant_conv_mapping)
        if self.verbose:
            print(f'Output shape: {decoded_images.shape}')

        return decoded_images, codebook_indices, q_loss

    def encode(self, x):
        encoded_images = self.encoder(x)
        quantized_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quantized_encoded_images)
        return codebook_mapping, codebook_indices

    def decode(self, z):
        quantized_codebook_mapping = self.post_quant_conv(z)
        decoded_images = self.decoder(quantized_codebook_mapping)
        return decoded_images


class vqvaeTraining():
    def __init__(self, args):

        self.verbose = args.verbose
        self.save_ckpt = args.save_ckpt
        self.save_losses = args.save_losses
        self.dataset = args.dataset
        
        self.vqvae = VQVAE(args)
        self.loader = loadData(args)

        self.create_ckpt(self.save_ckpt, self.save_losses)
        self.train(args)        

    @staticmethod
    def create_ckpt(ckpt_path, losses_path):
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        if not os.path.exists(losses_path):
            os.makedirs(losses_path)


    def train(self, args):

        train_dataset, val_dataset = self.loader.getDataloader()
        print(f'Training with {len(train_dataset)*args.batch_size} images')

        iterations_per_epoch = len(train_dataset)
        learning_rate = 3e-4
        criterion = torch.nn.MSELoss()
        opt_vq = torch.optim.Adam(
            list(self.vqvae.encoder.parameters())+
            list(self.vqvae.decoder.parameters())+
            list(self.vqvae.codebook.parameters())+
            list(self.vqvae.quant_conv.parameters())+
            list(self.vqvae.post_quant_conv.parameters()),
            lr=learning_rate,eps=1e-8, betas=(args.beta1, args.beta2))

        print('--> STARTING VQVAE <--')

        best_val_loss = 1e9
        patience_counter = 0
        patience = args.patience
        all_train_loss = []
        all_val_loss = []

        for epoch in range(args.epochs):
            epoch_train_losses = []
            epoch_val_losses = []            

            self.vqvae.train()
            for imgs, _ in tqdm(train_dataset):                    
                imgs = imgs.to(device=args.device)
                decoded_images, min_indices, q_loss = self.vqvae(imgs)

                rec_loss = criterion(imgs, decoded_images)
                vq_loss = rec_loss + q_loss

                opt_vq.zero_grad()
                vq_loss.backward(retain_graph=True)
                opt_vq.step()
                epoch_train_losses.append(vq_loss.cpu().detach().numpy())                        

            self.vqvae.eval()
            for val_imgs, _ in val_dataset:
                imgs = imgs.to(device=args.device)
                decoded_images, min_indices, q_loss = self.vqvae(imgs)
    
                rec_loss = criterion(imgs, decoded_images)
                vq_loss = rec_loss + q_loss
                epoch_val_losses.append(vq_loss.cpu().detach().numpy())
                    
            train_loss, val_loss = np.mean(epoch_train_losses), np.mean(epoch_val_losses)
            all_train_loss.append(train_loss)
            all_val_loss.append(val_loss)

            #Early Stopping
            if val_loss <  best_val_loss:
                best_val_loss = val_loss
                torch.save(self.vqvae.state_dict(), os.path.join(self.save_ckpt, f'vqvae_bestVal_{self.dataset}.pt'))
            else:
                patience_counter += 1

            #if patience_counter > patience:
            #    break

            if epoch % 10 == 0:
                np.save(os.path.join(self.save_losses, f'VQVAE_train_loss_{self.dataset}.npy'), all_train_loss)
                np.save(os.path.join(self.save_losses, f'VQVAE_val_loss_{self.dataset}.npy'), all_val_loss)
                
                self.vqvae.eval()
                for x, _ in train_dataset:
                    x = x.to(args.device)
                    decoded_x, a, b = self.vqvae(x)
                    img = decoded_x[0, : , :, :].permute(1, 2, 0).cpu().detach().numpy()
                    image = Image.fromarray(img.astype('uint8'),'RGB')
                    image.save('teste.png')
                    break


            if epoch % 50 == 0:
                torch.save(self.vqvae.state_dict(), os.path.join(self.save_ckpt, f'vqvae_epoch_{epoch}_{self.dataset}.pt'))
        torch.save(self.vqvae.state_dict(), os.path.join(self.save_ckpt, f'vqvae_lastEpoch_{self.dataset}.pt'))


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()

    #VQ_VAE ARGS
    parser.add_argument('--latent-dim', type=int, default=256)
    parser.add_argument('--num_res_blocks', type=int, default=4)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--num_codebook_vectors', type=int, default=256)
    parser.add_argument('--beta', type=float, default=0.25)
    parser.add_argument('--use_ema', type=bool, default=True)
    parser.add_argument('--learning-rate', type=float, default=2.25e-05)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.9)
    
    #DATASET ARGS
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--imagenetPath', type=str, default='/scratch2/pedroroblesduten/classical_datasets/imagenet')
    parser.add_argument('--imagenetTxtPath', type=str, default='/scratch2/pedroroblesduten/classical_datasets/imagenet/txt_files')
    parser.add_argument('--cifar10Path', type=str, default='/scratch2/pedroroblesduten/classical_datasets/cifar10')

    #TRAINING ARGS
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--patience', type=int, default=10)

    #PATH ARGS
    parser.add_argument('--save_ckpt', type=str, default='/scratch2/pedroroblesduten/MASKGIT/ckpt')
    parser.add_argument('--save_losses', type=str, default='/scratch2/pedroroblesduten/MASKGIT/losses')
    
    args = parser.parse_args()

    trainvqvae = vqvaeTraining(args)

