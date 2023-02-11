import torch
import torch.nn
import torch.nn.functional as F
import argparse
from maskgit_transformer import MaskGITTransformer, MaskGITconfig
import numpy as np
from vqvae_training import VQVAE
import math
#from inference import generateImages

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
    parser.add_argument('--save_results_path', type=str, default='/scratch2/pedroroblesduten/MASKGIT/results/')

    args = parser.parse_args()
    args.gpt_load_ckpt = None

    mask_token = 1024
    sos_token = 1025
    batch_size = 2
    non_mask_confidence = 1e9
    # args.verbose = True
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

    def maskSequence(seq, label=None):
        #TODO: available training for no label data in the same model
        z_indices = seq.view(x.shape[0], -1)

        sos_tokens = torch.ones(x.shape[0], 1, dtype=torch.long, device=z_indices.device) * sos_token

        r = math.floor(getMask(np.random.uniform(), 'cosine') * z_indices.shape[1])
        sample = torch.rand(z_indices.shape, device=z_indices.device).topk(r, dim=1).indices
        mask = torch.zeros(z_indices.shape, dtype=torch.bool, device=z_indices.device)
        mask.scatter_(dim=1, index=sample, value=True)

        masked_indices = mask_token * torch.ones_like(z_indices, device=z_indices.device)
        masked_indices = mask * z_indices + (~mask) * masked_indices


        masked_indices = torch.cat((sos_tokens, masked_indices), dim=1)
        target = torch.cat((sos_tokens, z_indices), dim=1)

        if label is not None:
            label_tokens = label * torch.ones([batch_size, 1], device=label.device)
            label_tokens = label_tokens + mask_token
            input_tokens = torch.concat([label_tokens, masked_indices], dim=-1)
            target_tokens = torch.concat([label_tokens, target], dim=-1)

        return input_tokens.long(), target_tokens.long()
    
    # GPT ARCHITECURE CONFIG PARAMETERS

    maskgitconf = MaskGITconfig(block_size = 258, # how far back does the model look? i.e. context size
                        vocab_size = 1036,
                        n_layers = 10,
                        n_heads = 8,
                        embedding_dim = 768, # size of the model
                        dropout = 0.# for determinism
                        )

    vq_vae = VQVAE(args).to('cuda')
    transformer = MaskGITTransformer(maskgitconf).to('cuda')
    
    x = torch.randn(batch_size, 3, 256, 256)
    x = x.to('cuda')
    label = torch.randint(9, (batch_size, 1))
    label = label.to('cuda')
    print('INPUT SHAPE: ', x.shape)
    print('LABEL SHAPE: ', label.shape)
    a, sequence, b = vq_vae(x)
    print(sequence.device)
    print('VQ_VAE OUTPUT SEQUENCE SHAPE: ',sequence.shape)
    inp_seq, tar_seq = maskSequence(sequence, label)
    print('INPUT SEQ SHAPE: ', inp_seq[0])
    print('TARGET SEQ SHAPE: ', tar_seq[0])
    print(f'MASKGIT CONFIGURATIONS: {maskgitconf}')
    logits = transformer(inp_seq)
    print('logits', logits.shape)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tar_seq.reshape(-1))
    print('loss: ' ,loss)
    #train_GPT = trainMaskGITTransformers(args, gptconf, run_vqvae=False)


    print('--- GENERATION ---')
    def maskByConfidence(probs, mask_len, temperature=1.0):
        #The add term is just some random noise
        confidence = torch.log(probs) + temperature * torch.distributions.gumbel.Gumbel(0, 1).sample(probs.shape).to(logits.device)
        sorted_confidence, _ = torch.sort(confidence, dim=-1)
        cut_off = torch.take_along_dim(sorted_confidence, mask_len.to(torch.long), dim=-1)
        masking = (confidence < cut_off)
        return masking

    def createInputTokensNormal(batch_size, label):
        blank_tokens = torch.ones((batch_size, 256), device=label.device)
        masked_tokens = mask_token * blank_tokens
        sos_tokens = torch.ones(batch_size, 1, dtype=torch.long, device=label.device) * sos_token

        if label is not None:
            label_tokens = label * torch.ones([batch_size, 1], device=label.device)
            label_tokens = label_tokens + mask_token
            masked_tokens = torch.cat((sos_tokens, masked_tokens), dim=1)
            masked_tokens = torch.concat([label_tokens, masked_tokens], dim=-1)
            print(masked_tokens)
        else:
            inputs = torch.cat((sos_tokens, masked_tokens), dim=1)
        
        return masked_tokens.to(torch.int32)
    def partialImageCreateInputTokens(idx, batch_size, label, start, end):
        #Create the input tokens passing just some part of the full sequence
        blank_tokens = torch.ones((batch_size, 256), device=label.device)
        masked_tokens = mask_token * blank_tokens
        masked_tokens[:, start:end] = idx[:, start:end]
        sos_tokens = torch.ones(inputs.shape[0], 1, dtype=torch.long, device=inputs.device) * sos_token

        if label is not None:
            label_tokens = label * torch.ones([batch_size, 1], device=label.device)
            label_tokens = label_tokens + mask_token
            masked_tokens = torch.cat((sos_tokens, label_tokens), dim=1)
            masked_tokens = torch.concat([label_tokens, masked_tokens], dim=-1)
        else:
            inputs = torch.cat((sos_tokens, masked_tokens), dim=1)
        
        return masked_tokens.to(torch.int32)
    def generateTokens(idx=None, label=None):

        #Getting inputs
        if idx is None:
            inputs = createInputTokensNormal(batch_size, label)
        else:
            inputs = partialImageCreateInputTokens(idx, batch_size, label, 40, 80)

        unknown_tokens_0 = torch.sum(inputs == mask_token, dim=-1)
        current_tokens = inputs
        for it in range(5):
            logits = transformer(inputs)
            pred_tokens = torch.distributions.categorical.Categorical(logits=logits).sample()
            unknown_tokens = (current_tokens == mask_token)
            sampled_tokens = torch.where(unknown_tokens, pred_tokens, current_tokens)

            r = 1. * (it + 1) / 5
            mask_ratio = getMask(r, 'cosine')
            
            probs = F.softmax(logits, dim=-1)
            selected_probs = torch.squeeze(torch.take_along_dim(probs, torch.unsqueeze(sampled_tokens, -1), -1), -1)

            selected_probs = torch.where(unknown_tokens, selected_probs, non_mask_confidence)

            mask_len = torch.unsqueeze(torch.floor(unknown_tokens_0 * mask_ratio), 1)  
            mask_len = torch.maximum(torch.ones_like(mask_len), torch.minimum(torch.sum(unknown_tokens, dim=-1, keepdim=True)-1, mask_len))

            #Noise for randoness
            masking = maskByConfidence(selected_probs, mask_len, temperature=4.5*(1.0 - r))

            current_tokens = torch.where(masking, mask_token, sampled_tokens)
        
        if label is None:
            generated_tokens = current_tokens[:, 1:]
        else:
            generated_tokens = current_tokens[:, 2:]

        return generated_tokens
    gen_tokens = generateTokens(idx=inp_seq, label=label)
    print('GENERATED TOKENS: ', gen_tokens)
