import torch
from torch import Tensor
import torch.nn as nn
from decoder import DecoderLayer

class GPT(nn.Module):
    def __init__(
        self,
        config
        ):

        super().__init__()
        self.word_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.window, config.d_model)
        self.decoder = nn.ModuleList([DecoderLayer(config) for _ in range(config.layers)])
        self.dropout = nn.Dropout(config.p)
        self.config = config

        nn.init.normal_(self.word_emb.weight, 0, 0.02)

    def forward(self, x: Tensor) -> Tensor:
        batch, window = x.shape
        positions = torch.arange(0, window).expand(batch, window).to(self.config.device) 
        dec_out = self.dropout(self.word_emb(x) + self.pos_emb(positions))

        for dec_layer in self.decoder:
            dec_out = dec_layer(dec_out)

        return dec_out