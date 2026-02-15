# 1. f(x)
from teenygrad import InterpretedTensor
a, b = InterpretedTensor.arange(12).reshape((3,4)), InterpretedTensor.arange(20).reshape((4,5))
c = a@b
print(c)

# 2. f'(x)
import torch
from teenygrad import InterpretedTensor
x_pt = torch.tensor(3.0, requires_grad=True)
y_pt = x_pt*x_pt
y_pt.backward()

# f:R->R       f':R->R
# f(x)=x^2 ==> f'(x)=2x
#   x =3   ==> f'(x)=6
print(x_pt.grad.item())

x = InterpretedTensor((1,), [3.0], requires_grad=True) #, requires_grad=True)
y = x * x
print(y)
y.backward()
print(x.grad)

# 3. dnn
# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# class MLP(nn.Module):
#   """
#   takes the previous block_size tokens, encodes them with a lookup table,
#   concatenates the vectors and predicts the next token with an MLP.

#   Reference:
#   Bengio et al. 2003 https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
#   """

#   def __init__(self, config):
#     super().__init__()
#     self.block_size = config.block_size
#     self.vocab_size = config.vocab_size
#     self.wte = nn.Embedding(config.vocab_size + 1, config.n_embd) # token embeddings table
#     # +1 in the line above for a special <BLANK> token that gets inserted if encoding a token
#     # before the beginning of the input sequence
#     self.mlp = nn.Sequential(
#       nn.Linear(self.block_size * config.n_embd, config.n_embd2),
#       nn.Tanh(),
#       nn.Linear(config.n_embd2, self.vocab_size)
#     )

#   def get_block_size(self): return self.block_size

#   def forward(self, idx, targets=None):
#     # gather the word embeddings of the previous 3 words
#     embs = []
#     for k in range(self.block_size):
#       tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)
#       idx = torch.roll(idx, 1, 1)
#       idx[:, 0] = self.vocab_size # special <BLANK> token
#       embs.append(tok_emb)

#     # concat all of the embeddings together and pass through an MLP
#     x = torch.cat(embs, -1) # (b, t, n_embd * block_size)
#     logits = self.mlp(x)

#     # if we are given some desired targets also calculate the loss
#     loss = None
#     if targets is not None: loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

#     return logits, loss