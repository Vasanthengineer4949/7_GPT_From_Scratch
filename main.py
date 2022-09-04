import torch
from config import Config
from gpt import GPT
from head import CLSHead

config = Config()
gpt = GPT(config)
cls_test = CLSHead(config, gpt)
cls_logits = cls_test(torch.randint(0, config.vocab_size, (1, config.window)))
print(cls_logits[0][0], cls_logits.shape)