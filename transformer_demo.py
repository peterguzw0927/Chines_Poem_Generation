import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import set_seed
set_seed(3407)
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

#enc = tiktoken.get_encoding("gpt2")#how to use tiktoken? further optimization

class CharDataset(Dataset):

    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))
        
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
    
    def __len__(self):
        return (len(self.data) - self.block_size) #//7

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


#--------------------------------------------------------------------------------------------------------------------------

block_size = 64

text = open('libai.txt','r',encoding='UTF-8').read()
train_dataset = CharDataset(text,block_size)


from mingpt.model import GPT

model_config = GPT.get_default_config()
model_config.model_type = 'gpt2'
model_config.vocab_size = train_dataset.vocab_size
model_config.block_size = train_dataset.block_size
model = GPT(model_config)


from mingpt.trainer import Trainer

train_config = Trainer.get_default_config()
train_config.learning_rate = 1e-4 
train_config.max_iters = 10000
train_config.num_workers = 0
trainer = Trainer(train_config, model, train_dataset)

def batch_end_callback(trainer):
    if trainer.iter_num % 100 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
trainer.set_callback('on_batch_end', batch_end_callback)

trainer.run()

model.eval()


#-----------------------------------------------------------------------------------------------------------------------

encode = lambda s: [train_dataset.stoi[c] for c in s]#encoder
decode = lambda l: ''.join([train_dataset.itos[i] for i in l])#decoder
num_samples = 5
steps = 128


prompt = "中秋"
x = torch.tensor(encode(prompt),dtype=torch.long).to(device='cuda:0')
x = x.expand(num_samples, -1)
y = model.generate(x,max_new_tokens=steps,do_sample=True,top_k=40)#long tensor [b,t], t->blocksize, b?

for i in range(num_samples):
        out = decode(y[i].cpu().numpy())
        print('-'*80)
        print(out)