import torch.nn as nn
from names_generator import CONFIG
from names_generator.head import MultiHeadAttention
from names_generator.feed_forward import FeedForward


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = CONFIG.num_embed // CONFIG.num_head
        self.sa = MultiHeadAttention(CONFIG.num_head, head_size, CONFIG.num_embed, CONFIG.context_length, CONFIG.dropout)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(CONFIG.num_embed)
        self.ln2 = nn.LayerNorm(CONFIG.num_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
