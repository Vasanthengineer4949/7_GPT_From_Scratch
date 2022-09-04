import torch
import torch.nn as nn
from torch import Tensor

class ClassificationHead(nn.Module):
    def __init__(
        self,
        config,
        gpt
        ):
        super().__init__()
        self.gpt = gpt
        self.classifier = nn.Linear(config.d_model, config.n_class)
        nn.init.normal_(self.classifier.weight, std=0.02)
        
    def forward(self, x: Tensor) -> Tensor:
        dec_out = self.gpt(x)
        cls_logits = self.classifier(dec_out)
        return cls_logits