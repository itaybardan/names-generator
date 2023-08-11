import torch
import torch.nn as nn
from torch.functional import F

from names_generator import CONFIG
from names_generator.block import Block


class NamesGeneratorModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(vocab_size, CONFIG.num_embed)
        self.position_embedding_table = nn.Embedding(CONFIG.context_length, CONFIG.num_embed)
        self.blocks = nn.Sequential(*[Block() for _ in range(CONFIG.num_layers)])
        self.ln_f = nn.LayerNorm(CONFIG.num_embed)
        self.lm_head = nn.Linear(CONFIG.num_embed, vocab_size)

        def init_weights(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        self.apply(init_weights)

    def forward(self, input_data, targets=None):
        _, time = input_data.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(input_data)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(time, device=CONFIG.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            batch, time, character = logits.shape
            logits = logits.view(batch * time, character)
            targets = targets.view(batch * time)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, block_size, max_tokens):
        for _ in range(max_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probabilities = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probabilities, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
