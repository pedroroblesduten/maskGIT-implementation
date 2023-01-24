from encoder_decoder import Encoder, Decoder
from torch.utils.data import DataLoader
from codebook import CodebookEMA
import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent-dim', type=int, default=256)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--verbose', type=str, default=True)
    parser.add_argument('--num_codebook_vectors', type=int, default=1024)
    parser.add_argument('--beta', type=float, default=0.25)
    args = parser.parse_args()

    batch_size = 1
    n_images = 6

    data = torch.stack([torch.randn(3, 256, 256) for i in range(n_images)])
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    encoder = Encoder(args).to('cuda')
    decoder = Decoder(args).to('cuda')
    codebook = CodebookEMA(args).to('cuda')
    for img in dataloader:
        img = img.to('cuda')
        enc_out = encoder(img)
        print(enc_out.shape)
        dec, ind, loss = codebook(enc_out)
        print(dec.shape)
        dec_out = decoder(enc_out)
        print(dec_out.shape)

