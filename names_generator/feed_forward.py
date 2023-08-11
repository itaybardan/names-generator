import torch.nn as nn
from names_generator import CONFIG


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CONFIG.num_embed, 4 * CONFIG.num_embed),
            nn.ReLU(),
            nn.Linear(4 * CONFIG.num_embed, CONFIG.num_embed),
            nn.Dropout(CONFIG.dropout),
        )

    def forward(self, x):
        return self.net(x)
