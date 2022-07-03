# movie_sentiment.py

# simple sentiment analysis using PyTorch EmbeddingBag
# Anaconda3-2020.02 (Python 3.7.6)
# PyTorch 1.8.0-CPU TorchText 0.9
# Windows 10

import numpy as np
import torch as T
import torchtext as tt
import collections

device = T.device("cpu")

# -----------------------------------------------------------

# data file looks like:
# 0 ~ This was a BAD movie.
# 1 ~ I liked this film! Highly recommeneded.
# 0 ~ Just awful
# . . .


def make_vocab(fn):
  # create Vocab object to convert words/tokens to IDs
  # assumes an instantiated global tokenizer
  # toker = tt.data.utils.get_tokenizer("basic_english")
  counter_obj = collections.Counter()
  f = open(fn, "r")
  for line in f:
    line = line.strip()
    txt = line.split("~")[1]
    # print(txt); input()
    split_and_lowered = g_toker(txt)  # global
    counter_obj.update(split_and_lowered)
  f.close()
  result = tt.vocab.Vocab(counter_obj, min_freq=1,
                          specials=('<unk>', '<pad>'))
  return result  # a Vocab object


# globals are needed for the collate_fn() function
g_toker = tt.data.utils.get_tokenizer("basic_english")
g_vocab = make_vocab(".\\Data\\reviews20.txt")


def make_data_list(fn):
  # get all data into one big list of (label, review) tuples
  # result will be passed to DataLoader, used by collate_fn
  result = []
  f = open(fn, "r")
  for line in f:
    line = line.strip()
    parts = line.split("~")
    tpl = (parts[0], parts[1])  # label, review
    result.append(tpl)
  f.close()
  return result

# -----------------------------------------------------------


def collate_data(batch):
  # rearrange a batch and compute offsets too
  # needs a global vocab and tokenizer
  label_lst, review_lst, offset_lst = [], [], [0]
  for (_lbl, _rvw) in batch:
    label_lst.append(int(_lbl))  # string to int

    rvw_idxs = [g_vocab.stoi[tok] for tok in g_toker(_rvw)]  # idxs
    rvw_idxs = [g_vocab[tok] for tok in g_toker(_rvw)]  # stoi opt.
    rvw_idxs = T.tensor(rvw_idxs, dtype=T.int64)  # to tensor
    review_lst.append(rvw_idxs)
    offset_lst.append(len(rvw_idxs))

  label_lst = T.tensor(label_lst, dtype=T.int64).to(device)
  # print(offset_lst); input()
  offset_lst = T.tensor(offset_lst[:-1]).cumsum(dim=0).to(device)
  # print(offset_lst); input()
  review_lst = T.cat(review_lst).to(device)  # 2 tensors to 1

  return (label_lst, review_lst, offset_lst)

# -----------------------------------------------------------


class NeuralNet(T.nn.Module):

  def __init__(self):
    super(NeuralNet, self).__init__()
    self.vocab_size = len(g_vocab)
    self.embed_dim = 50
    self.num_class = 2

    self.embed = T.nn.EmbeddingBag(self.vocab_size,
                                   self.embed_dim)
    self.fc1 = T.nn.Linear(self.embed_dim, 20)
    self.fc2 = T.nn.Linear(20, self.num_class)

    lim = 0.05
    self.embed.weight.data.uniform_(-lim, lim)
    self.fc1.weight.data.uniform_(-lim, lim)
    self.fc1.bias.data.zero_()
    self.fc2.weight.data.uniform_(-lim, lim)
    self.fc2.bias.data.zero_()

  def forward(self, reviews, offsets):
    z = self.embed(reviews, offsets)
    z = T.tanh(self.fc1(z))  # tanh activation
    z = self.fc2(z)  # no activation: CrossEntropyLoss
    return z

# -----------------------------------------------------------


def train(net, ldr, bs, me, le, lr):
  # network, loader, bat_size, max_epochs, log_every, lrn_rate
  net.train()
  opt = T.optim.SGD(net.parameters(), lr=lr)
  loss_func = T.nn.CrossEntropyLoss()  # will apply softmax
  print("\nStarting training")
  for epoch in range(0, me):
    epoch_loss = 0.0
    for bix, (labels, reviews, offsets) in enumerate(ldr):
      opt.zero_grad()
      oupt = net(reviews, offsets)  # get predictions
      loss_val = loss_func(oupt, labels)  # compute loss
      loss_val.backward()  # compute gradients
      epoch_loss += loss_val.item()  # accum loss for display
      opt.step()  # update net weights
    if epoch % le == 0:
      print("epoch = %4d   loss = %0.4f" % (epoch, epoch_loss))
  print("Done ")

# -----------------------------------------------------------


def accuracy(net, meta_lst):
  net.eval()
  ldr = T.utils.data.DataLoader(meta_lst,
                                batch_size=1, shuffle=False, collate_fn=collate_data)
  num_correct = 0
  num_wrong = 0
  for bix, (labels, reviews, offsets) in enumerate(ldr):
    with T.no_grad():
      oupt = net(reviews, offsets)  # get prediction values
    pp = T.softmax(oupt, dim=1)  # pseudo-probability
    predicted = T.argmax(pp, dim=1)  # 0 or 1 as tensor
    if labels.item() == predicted.item():
      num_correct += 1
    else:
      num_wrong += 1

  return (num_correct * 1.0) / (num_correct + num_wrong)

# -----------------------------------------------------------


def main():
  # 0. get started
  print("\nBegin PyTorch EmbeddingBag movie sentiment ")
  T.manual_seed(2)
  np.random.seed(2)

  # 1. create training DataLoader object
  print("\nTraining data looks like: ")
  print("0 ~ This was a BAD movie.")
  print("1 ~ I liked this film! Highly recommeneded.")
  print("0 ~ Just awful")
  print(" . . . ")

  bat_size = 3
  print("\nLoading training data into meta-list ")
  data_lst = make_data_list(".\\Data\\reviews20.txt")
  print("Creating DataLoader from meta-list ")
  train_ldr = T.utils.data.DataLoader(data_lst,
                                      batch_size=bat_size, shuffle=True,
                                      collate_fn=collate_data)

  # 2. create neural net
  print("\nCreating an EmbeddingBag neural net ")
  net = NeuralNet().to(device)

  # 3. train movie sentiment model
  max_epochs = 300
  log_interval = 30
  lrn_rate = 0.05

  print("\nbat_size = %3d " % bat_size)
  print("max epochs = " + str(max_epochs))
  print("loss = CrossEntropyLoss")
  print("optimizer = SGD")
  print("lrn_rate = %0.3f " % lrn_rate)
  train(net, train_ldr, bat_size, max_epochs,
        log_interval, lrn_rate)

  # 4. compute model classification accuracy
  acc_train = accuracy(net, data_lst)
  print("\nAccuracy of model on training data = \
%0.4f " % acc_train)

  # 5. TODOs: test data accuracy, save model

  # 6. make a prediction on a new review
  print("\nNew movie review: Overall, I liked the film.")
  review_lst = [("-1", "Overall, I liked the film.")]
  ldr = T.utils.data.DataLoader(review_lst,
                                batch_size=1, shuffle=True, collate_fn=collate_data)
  net.eval()
  (_, review, offset) = iter(ldr).next()
  with T.no_grad():
    oupt = net(review, offset)  # get raw prediction values
  pp = T.softmax(oupt, dim=1)   # as pseudo-probabilities
  print("Sentiment prediction probabilities [neg, pos]: ")
  print(pp)

  print("\nEnd demo ")


if __name__ == "__main__":
  main()

# -----------------------------------------------------------

# copy,paste, remove comment chars, save as reviews20.txt
# 0 ~ This was a BAD movie.
# 1 ~ I liked this film! Highly recommeneded.
# 0 ~ Just awful
# 1 ~ Good film, acting
# 0 ~ Don't waste your time - A real dud
# 0 ~ Terrible
# 1 ~ Great movie.
# 0 ~ This was a waste of talent.
# 1 ~ I liked this movie a lot. See it.
# 1 ~ Best film I've seen in years.
# 0 ~ Bad acting and a weak script.
# 1 ~ I recommend this film to everyone
# 1 ~ Entertaining and fun.
# 0 ~ I didn't like this movie at all.
# 1 ~ A good old fashioned story
# 0 ~ The story made no sense to me.
# 0 ~ Amateurish from start to finish.
# 1 ~ I really liked this move. Lot of fun.
# 0 ~ I disliked this movie and walked out.
# 1 ~ A thrilling adventure for all ages.
