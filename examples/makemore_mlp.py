def build_dataset(t):
  import random

  words = open('./data/names.txt', 'r').read().splitlines()
  v = sorted(list(set(''.join(words))))
  encode = { c:i+1 for i,c in enumerate(v) }
  encode['.'] = 0
  decode = { i:c for c,i in encode.items() }

  def gen_dataset(words, t):
    X, Y = [], []
    for w in words:
      context = [0] * t
      for c in w + '.':
        X.append(context)
        Y.append(encode[c])
        # print(''.join(decode[i] for i in context), '-->', decode[encode[c]])
        context = context[1:] + [encode[c]]
    X, Y = pg.tensor(X), pg.tensor(Y) # X:(N,C) Y:(N)
    return X, Y

  random.seed(42)
  random.shuffle(words)
  n1, n2 = int(0.8*len(words)), int(0.9*len(words))
  Xtraining, Ytraining = gen_dataset(words[:n1], t)
  Xdev, Ydev = gen_dataset(words[n1:n2], t)
  Xte, Yte = gen_dataset(words[n2:], t)
  return Xtraining, Ytraining

class MLP():
  """
  model: Neural Language Models (Bengio et al. 2003) URL: https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
  key:
  b: batch size, t: sequence length
  v: vocabulary size, e: dimension of embedding, d: dimension of model
  """
  
  def __init__(self, cfg):
    super().__init__()
    b, t, v, e, d = cfg.b, cfg.t, cfg.v, cfg.e, cfg.d
    self.wte = nn.Embedding(v+1, e) # token embeddings table (+1 for <BLANK>)
    l1 = nn.Linear(t*e, d, b=False)
    l2 = nn.Linear(d, d, b=False)
    l3 = nn.Linear(d, v, b=False)

  def forward(self, i, targets=None):
    embs = [] # gather the word embeddings of the previous 3 words
    for k in range(self.b):
      tok_emb = self.wte(i) # token embeddings of shape (b, t, e)
      i = pg.roll(i, 1, 1)
      i[:, 0] = self.v # special <BLANK> token
      embs.append(tok_emb)

    # concat all of the embeddings together and pass through an MLP
    x = pg.cat(embs, -1) # (b, t, e * block_size)
    x = self.l1(x).tanh()
    x = self.l2(x).tanh()
    x = self.l3(x)
    yhat = x

    # if we are given some desired targets also calculate the loss
    loss = None
    if targets is not None: loss = nn.losses.cross_entropy(yhat.view(-1, yhat.size(-1)), targets.view(-1), ignore_index=-1)
    return yhat, loss
  
if __name__ == "__main__":
  b, t, v, e, d = 32, 3, 27, 10, 200 # init hyperparameters
  X, Y = build_dataset(t) # init data
  C = pg.randn((v,e), generator=g) # init embedding               move ?
  model = MLP() # init model
  params = [C] + [p for l in model for p in l.parameters()]     # move ?
  for p in params: p.requires_grad = True                       # move ?

  N, losses, steps = X.shape[0], [], [] # train
  for step in range(200000):
    i_b = pg.randint(0, N, (b,))
    X_b, Y_b = X[i_b], Y[i_b]
    X_bd = C[X_b].view(-1, t * e) # 0. embed
    for layer in model: X_bd = layer(X_bd) # 1. forward

    loss = X_bd.cross_entropy(Y_b)
    for layer in model: layer.out.retain_grad() # move?
    for p in params: p.grad = None
    loss.backward() # 2. backward

    for p in params: p.data += -0.01 * p.grad # 3. update
    # optimizer.step()?

    steps.append(step)
    losses.append(loss.log10().item())
    if step % 10000 == 0: print(f"step: {step}/{200000}, loss {loss.item()}")

    plt.plot(steps, losses)