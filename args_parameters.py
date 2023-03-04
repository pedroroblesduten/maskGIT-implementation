import argparse
from maskgit_transformer import MaskGITconfig

def getArgs():
    num_code_vectors = 256
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    #VQ_VAE ARGS
    args.latent_dim = int(256)
    args.num_res_blocks = int(4)
    args.verbose = False
    args.num_codebook_vectors = int(256)
    args.beta = 0.25
    args.use_ema = True
    args.learning_rate = 2.25e-05
    args.beta1 = 0.5
    args.beta2 = 0.9
    
    #DATASET ARGS
    args.dataset = 'CIFAR10'
    args.imagenetPath = '/scratch2/pedroroblesduten/classical_datasets/imagenet'
    args.imagenetTxtPath = '/scratch2/pedroroblesduten/classical_datasets/imagenet/txt_files'
    args.cifar10Path = '/scratch2/pedroroblesduten/classical_datasets/cifar10'

    #TRAINING ARGS
    args.epochs = 200
    args.batch_size = 48
    args.device= 'cuda'
    args.patience = 10

    #PATH ARGS
    args.save_ckpt = '/scratch2/pedroroblesduten/MASKGIT/ckpt'
    args.save_losses = '/scratch2/pedroroblesduten/MASKGIT/losses'
    args.vqvae_load_ckpt = '/scratch2/pedroroblesduten/MASKGIT/ckpt/vqvae_bestVal_CIFAR10.pt'
    args.gpt_load_ckpt = '/scratch2/pedroroblesduten/MASKGIT/ckpt/MASKGIT_bestVAL.pt'
    #args.gpt_load_ckpt = None
    args.save_results_path = '/scratch2/pedroroblesduten/MASKGIT/results/'
    
    #TRANSFORMERS ARGS
    args.mask_token = 256
    args.sos_token = 256+1
    args.gen_iter = 8
    args.transformer_epochs = 300

    return args

def getConfig():
    transformerConfig = MaskGITconfig(block_size = 66,
                                      vocab_size = 268,
                                      n_layers = 16,
                                      n_heads = 12,
                                      embedding_dim = 768,
                                      dropout = 0.
                                      )
    return transformerConfig

