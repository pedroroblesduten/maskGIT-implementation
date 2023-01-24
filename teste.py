from encoder_decoder import Encoder, Decoder
from torch.utils.data import DataLoader
import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent-dim', type=int, default=256)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    args = parser.parse_args()

    batch_size = 1
    n_images = 6

    data = torch.stack([torch.randn(3, 256, 256) for i in range(n_images)])
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    encoder = Encoder(args).to('cuda')
    decoder = Decoder(args).to('cuda')
    for img in dataloader:
        img = img.to('cuda')
        enc_out = encoder(img)
        print(enc_out.shape)
        dec_out = decoder(enc_out)

