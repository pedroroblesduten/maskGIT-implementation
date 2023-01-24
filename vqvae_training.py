import torch
import torch.nn as nn


class VQVAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.verbose = args.verbose
        self.use_ema = args.use_ema

        self.encoder = Encoder3D(args, verbose=self.verbose).to(device=args.device)
        self.decoder = Decoder3D(args, verbose=self.verbose).to(device=args.device)
        # if self.use_ema:
        self.codebook = CodebookEMA(args, verbose=self.verbose).to(device=args.device)
        #else:
        #    self.codebook = Codebook3D(args, verbose=self.verbose).to(device=args.device)
        self.quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device)
        self.post_quant_conv = nn.Conv32(args.latent_dim, args.latent_dim, 1).to(device=args.device)

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
            print(f'Output shape: {decoded_images.shape})

        return decoded_images, codebook_indices, q_loss
