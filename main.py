import torch
from config import Config
from gpt import GPT
from head import ClassificationHead

config = Config()
gpt = GPT(config)
classifier = ClassificationHead(config, gpt)
output = classifier(torch.randint(0, config.vocab_size, (1, config.window)))
print(output[0][0], output.shape)