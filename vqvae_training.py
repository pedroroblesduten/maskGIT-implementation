import torch
import torch.nn as nn
from load_data import loadData
from encoder_decoder import Encoder, Decoder
from codebook import CodebookEMA


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
    def __init__(self, args, verbose=False):

        self.verbose = verbose
        self.save_ckpt = args.vqvae_save_ckpt
        self.mri_vqvae = VQVAE(args)
        self.create_ckpt(self.save_ckpt)
        self.train(args)        

    @staticmethod
    def create_ckpt(ckpt_path):
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

    def train(self, args, verbose=False):
        train_dataset , val_dataset, _ = loadData(args)

        iterations_per_epoch = len(train_dataset)
        criterion = torch.nn.MSELoss()
        opt_vq = optim.Adam(
            list(self.mri_vqvae.encoder.parameters())+
            list(self.mri_vqvae.decoder.parameters())+
            list(self.mri_vqvae.codebook.parameters())+
            list(self.mri_vqvae.quant_conv.parameters())+
            list(self.mri_vqvae.post_quant_conv.parameters()),
            lr=args.learning_rate,eps=1e-8, betas=(args.beta1, args.beta2))

        print('--> STARTING VQVAE <--')

        best_val_loss = 1e9
        patience_counter = 0
        patience = args.patience
        all_train_loss = []
        all_val_loss = []

        for epoch in range(args.epochs):
            epoch_train_losses = []
            epoch_val_losses = []            

            self.mri_vqvae.train()
            for imgs in tqdm(train_dataset):                    
                imgs = imgs.to(device=args.device)
                decoded_images, min_indices, q_loss = self.mri_vqvae(imgs)

                rec_loss = criterion(imgs, decoded_images)
                vq_loss = rec_loss + q_loss

                opt_vq.zero_grad()
                vq_loss.backward(retain_graph=True)
                opt_vq.step()
                epoch_train_losses.append(vq_loss.cpu().detach().numpy())                        

            self.mri_vqvae.eval()
            for val_imgs in val_dataset:
                imgs = imgs.to(device=args.device)
                decoded_images, min_indices, q_loss = self.mri_vqvae(imgs)
    
                rec_loss = criterion(imgs, decoded_images)
                vq_loss = rec_loss + q_loss
                epoch_val_losses.append(vq_loss.cpu().detach().numpy())
                    
            train_loss, val_loss = np.mean(epoch_train_losses), np.mean(epoch_val_losses)
            all_train_loss.append(train_loss)
            all_val_loss.append(val_loss)

            #Early Stopping
            if val_loss <  best_val_loss:
                best_val_loss = val_loss
                torch.save(self.mri_vqvae.state_dict(), os.path.join(args.vqvae_save_ckpt, 'mri_vqvae_bestVal.pt'))
            else:
                patience_counter += 1

            #if patience_counter > patience:
            #    break

            if epoch % 10 == 0:
                np.save(os.path.join(args.save_loss, 'VQVAE_train_loss.npy'), all_train_loss)
                np.save(os.path.join(args.save_loss, 'VQVAE_val_loss.npy'), all_val_loss)
                torch.save(self.mri_vqvae.state_dict(), os.path.join(args.vqvae_save_ckpt, f'mri_vqvae_epoch_{epoch}.pt'))
        torch.save(self.mri_vqvae.state_dict(), os.path.join(args.save_ckpt, 'mri_vqvae_lastEpoch.pt'))


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()

    #VQ_VAE ARGS
    parser.add_argument('--latent-dim', type=int, default=256)
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--verbose', type=str, default=True)
    parser.add_argument('--num_codebook_vectors', type=int, default=1024)
    parser.add_argument('--beta', type=float, default=0.25)

    #TRAINING ARGS
    parser.add_argument('--epochs', type=int, default=1024)

    #PATH ARGS
    parser.add_argument('--save_ckpt', type=str, default='/scratch2/pedroroblesduten/MASKGIT/ckpt')
    parser.add_argument('--save_loss', type=str, default='/scratch2/pedroroblesduten/MASKGIT/losses')
    
    args = parser.parse_args()

