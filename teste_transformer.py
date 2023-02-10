import torch
import torch.nn
import torch.nn.functional as F
import argparse
from my_minGPT import GPT, GPTconfig
from maskgit_transformer import MaskGITTransformer, MaskGITconfig
import numpy as np
from vqvae_training import VQVAE
import math
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GPT_TRAINING")
    parser.add_argument('--gpt_batch_size', type=str, default=2)
    parser.add_argument('--gpt_epochs', type=str, default=500)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--latent-dim', type=int, default=256)
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--verbose', type=str, default=True)
    parser.add_argument('--num_codebook_vectors', type=int, default=1024)
    parser.add_argument('--beta', type=float, default=0.25)
    parser.add_argument('--use_ema', type=bool, default=True)
    parser.add_argument('--device', type=str, default='cuda')

    #TRAINING ARGS
    parser.add_argument('--epochs', type=int, default=1024)

    #PATH ARGS
    parser.add_argument('--save_ckpt', type=str, default='/scratch2/pedroroblesduten/MASKGIT/ckpt')
    parser.add_argument('--save_loss', type=str, default='/scratch2/pedroroblesduten/MASKGIT/losses')

    args = parser.parse_args()
    args.gpt_load_ckpt = None
    # args.verbose = True
    
    # GPT ARCHITECURE CONFIG PARAMETERS

    maskgitconf = GPTconfig(block_size = 257, # how far back does the model look? i.e. context size
                        vocab_size = 1026,
                        n_layers = 10,
                        n_heads = 8,
                        embedding_dim = 768, # size of the model
                        dropout = 0.# for determinism
                        )

    def getMask(seq, mode):
        ratio = np.random.uniform()
        print(ratio)
        if mode == "linear":
            return 1 - ratio
        elif mode == "cosine":
            return np.cos(ratio * np.pi / 2)
        elif mode == "square":
            return 1 - ratio ** 2
        elif mode == "cubic":
            return 1 - ratio ** 3

    vq_vae = VQVAE(args)
    transformer = MaskGITTransformer(maskgitconf).to('cuda')
    
    x = torch.randn(2, 3, 256, 256)
    x = x.to('cuda')
    print(x.shape)

    _, z_indices = vq_vae.encode(x)
    z_indices = z_indices.view(x.shape[0], -1)
    print(z_indices.shape)

    sos_tokens = torch.ones(x.shape[0], 1, dtype=torch.long, device=z_indices.device) * 1024

    r = math.floor(getMask(np.random.uniform(), 'cosine') * z_indices.shape[1])
    sample = torch.rand(z_indices.shape, device=z_indices.device).topk(r, dim=1).indices
    mask = torch.zeros(z_indices.shape, dtype=torch.bool, device=z_indices.device)
    mask.scatter_(dim=1, index=sample, value=True)
    print('MASK', mask.shape)
    print(mask)

    masked_indices = 1025 * torch.ones_like(z_indices, device=z_indices.device)
    a_indices = mask * z_indices + (~mask) * masked_indices

    a_indices = torch.cat((sos_tokens, a_indices), dim=1)

    target = torch.cat((sos_tokens, z_indices), dim=1)

    
    print('indice mask', a_indices.shape)
    print('targer', target.shape)
    logits = transformer(a_indices)
    print('logits', logits.shape)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
    print(loss)
    print(f'MASKGIT CONFIGURATIONS: {maskgitconf}')
    #train_GPT = trainMaskGITTransformers(args, gptconf, run_vqvae=False)
