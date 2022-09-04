import torch
from torch import Tensor
import torch.nn as nn
import math
import torch.nn.functional as F
from config import Config

class DecoderLayer(nn.Module):
    def __init__(
        self,
        config
        ):

        super().__init__()
        self.C = nn.Linear(config.d_model, config.d_model*3)
        self.linear = nn.Linear(config.d_model, config.d_model)
        self.FF = nn.Sequential(
            nn.Linear(config.d_model, config.inner_state),
            nn.GELU(),
            nn.Linear(config.inner_state, config.d_model),
            nn.Dropout(config.p)
        )
        nn.init.normal_(self.FF[0].weight, 0, 0.02)
        nn.init.normal_(self.FF[2].weight, 0, 0.02)
        self.LN1 = nn.LayerNorm(config.d_model)
        self.LN2 = nn.LayerNorm(config.d_model)
        self.head_dim = config.d_model // config.heads
        self.heads = config.heads
        self.dropout = nn.Dropout(config.p)

        

    def forward(self, x: Tensor) -> Tensor:
        batch, window, d = x.shape
        mask = self._make_mask(batch, window)
        
        c = self.C(x)
        q, k, v = torch.split(tensor=c, split_size_or_sections=d, dim=2)
        q = q.reshape(batch, window, self.heads, self.head_dim)
        k = k.reshape(batch, window, self.heads, self.head_dim)
        v = v.reshape(batch, window, self.heads, self.head_dim)

        QK = torch.einsum("bqhd, bkhd -> bhqk", [q, k]) / math.sqrt(d)
        QK = QK.masked_fill(mask==0, float("-inf"))
        scores = self.dropout(F.softmax(QK, dim=3))
        output = torch.einsum("bhqk, bvhd -> bqhd", [scores, v])
        concat = output.reshape(batch, window, d)
        linear = self.dropout(self.linear(concat))

        addnorm1 = self.LN1(x + linear)
        addnorm2 = self.LN2(addnorm1 + self.FF(addnorm1))
        return addnorm2

    def _make_mask(self, batch, window):
        mask = torch.tril(torch.ones((window, window)))
        return mask.reshape(batch, 1, window, window)