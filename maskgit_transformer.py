# Architecture from original JAX implementation from Google Research, but here in PyTorch
# https://github.com/google-research/maskgit/blob/main/maskgit/nets/maskgit_transformer.py

# Based on Andrej Karpathy GPT implementations:
# minGPT: https://github.com/karpathy/minGPT
# nanoGPT: https://github.com/karpathy/nanoGPT

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from dataclasses import dataclass

class AttentionHead(nn.Module):
    def __init__(self, embedding_dim, head_dim, dropout, block_size):
        super().__init__()

        self.linear_Q = nn.Linear(embedding_dim, head_dim)
        self.linear_K = nn.Linear(embedding_dim, head_dim)
        self.linear_V = nn.Linear(embedding_dim, head_dim)
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attn(self, query, key, value):
        B, T, d_k = query.size()
        scores = torch.bmm(query, key.transpose(1, 2))/sqrt(d_k)
        att_matrix = F.softmax(scores, dim=-1)
        att_matrix = self.dropout(att_matrix)
        output = att_matrix.bmm(value)
        return output

    def forward(self, x):
        one_head_attn = self.scaled_dot_product_attn(
            self.linear_Q(x),
            self.linear_K(x),
            self.linear_V(x),
        )
        return one_head_attn

class MultiHeadAttn(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embedding_dim % config.n_heads == 0
        self.block_size = config.block_size
        self.embedding_dim = config.embedding_dim
        self.n_heads = config.n_heads
        self.head_dim = config.embedding_dim//config.n_heads
        self.attn_heads = nn.ModuleList([
            AttentionHead(self.embedding_dim, self.head_dim, config.dropout, self.block_size)
            for _ in range(self.n_heads)
        ])
        self.output_linear = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = torch.cat([one_head(x) for one_head in self.attn_heads], dim=-1)
        x = self.resid_dropout(self.output_linear(x))
        return x

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.embedding_dim, 4*config.embedding_dim)
        self.linear2 = nn.Linear(4*config.embedding_dim, config.embedding_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x
        
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.embedding_dim)
        self.layer_norm2 = nn.LayerNorm(config.embedding_dim)
        self.attention = MultiHeadAttn(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):        
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x

class MlmLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.layer_norm(x)
        return x     


# ---- TRANSFORMERS CONFIGURATIONS ----
@dataclass
class MaskGITconfig:
    block_size: int = 1025
    vocab_size: int = 1025
    n_layers: int = 12
    n_heads: int = 12
    embedding_dim: int = 768
    dropout: float = 0.1


# --- THE MODEL ---
class MaskGITTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pos_emb = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.tk_emb = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.layernorm = nn.LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList()
        for _ in range(config.n_layers):
            self.blocks.append(Block(config))
        self.mlmlayer = MlmLayer(config)
        self.bias = nn.Parameter(torch.zeros(config.block_size, config.vocab_size))

    def forward(self, x):
        b, t = x.size()
        pos = torch.arange(0, t, dtype=torch.long, device=x.device).unsqueeze(0)

        token_embedding = self.tk_emb(x)
        positional_embedding = self.pos_emb(pos)

        x = self.dropout(self.layernorm(token_embedding + positional_embedding))

        for block in self.blocks:
            x = block(x)

        x = self.mlmlayer(x)
        
        logits = torch.matmul(x, self.tk_emb.weight.T) + self.bias

        return logits

    #From nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
    def configure_optimizers(self, weight_decay, learning_rate, betas):

        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer
