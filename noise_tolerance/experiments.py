from itertools import product
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import numpy.random as rn
import matplotlib.pyplot as plt

import joblib

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import ndcg_score, roc_auc_score, average_precision_score, dcg_score
from sklearn.linear_model import LogisticRegressionCV

import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm

class RanknetLoss():
  def __call__(self, out, y):
    losses = []
    for i in range(y.shape[1]):
      out_rel = out[y[:, i] == 1][:, i]
      out_irrel = out[y[:, i] == 0][:, i]
      out_diff = (out_rel.unsqueeze(0) - out_irrel.unsqueeze(1)).reshape(-1)
      losses.append(F.binary_cross_entropy_with_logits(out_diff, torch.ones_like(out_diff)))
    return torch.mean(torch.stack(losses))

class SymBCEWithLogitsLoss():
  def __call__(self, out, y):
    return -(torch.sigmoid(out) * y + (1 - torch.sigmoid(out)) * (1 - y)).mean()

class SymRanknetLoss():
  def __call__(self, out, y):
    losses = []
    for i in range(y.shape[1]):
      out_rel = out[y[:, i] == 1][:, i]
      out_irrel = out[y[:, i] == 0][:, i]
      out_diff = (out_rel.unsqueeze(0) - out_irrel.unsqueeze(1)).reshape(-1)
      losses.append(SymBCEWithLogitsLoss()(out_diff, torch.ones_like(out_diff)))
    return torch.mean(torch.stack(losses))

class Estimator():
  def __init__(self, *, loss=None):
    self.loss = loss
    self.model_dir = Path('models')
    self.model_stats = None

  def _fit(self, X, y, options, test=None):
    model = Model(X.shape[1], y.shape[1])
    run_name = str(hash((self.loss, *sorted(options.items(), key=lambda pair: pair[0]))))
    run_dir = self.model_dir / run_name
    run_dir.mkdir(exist_ok=True)
    max_epochs = options['max_epochs']
    weight_decay = options['weight_decay']
    lr = options['lr']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = {'logloss': nn.BCEWithLogitsLoss,
               'ranknet': RanknetLoss,
               'sym_logloss': SymBCEWithLogitsLoss,
               'sym_ranknet': SymRanknetLoss
               }[self.loss]()
    selection_lookup = {
      'auc': roc_auc_score,
      'ndcg@10': lambda a, b: ndcg_score(a, b, k=10),
      'map': average_precision_score,
      'loss': lambda a, b: -loss_fn(torch.tensor(b), torch.tensor(a)).item()
    }
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    train_X, val_X, train_y, val_y = train_test_split(X, y)
    val_perfs = []
    print(options)
    for i in tqdm(range(max_epochs)):
      optimizer.zero_grad()
      out = model(train_X)
      loss = loss_fn(out, train_y)
      loss.backward()
      optimizer.step()
      joblib.dump(model, run_dir / '{}.pkl'.format(i))
      val_perfs.append({name: fn(val_y.numpy(), model(val_X).detach().numpy())
                        for name, fn in selection_lookup.items()})
      if i % 10 == 0:
        print('training loss: {:.4f}, val: {}'.format(
          loss.item(),
          str({k: '{:.4f}'.format(v) for k, v in val_perfs[-1].items()})
        ))
        if test:
          test_X, test_y = test
          test_X = torch.tensor(test_X, dtype=torch.float32)
          test_perf = ['{}: {:.4f}'.format(name, fn(np.array(test_y), model(test_X).detach().numpy()))
                       for name, fn in selection_lookup.items()]
          print('test perf: {}'.format(' '.join(test_perf)))
    val_perfs = pd.DataFrame(val_perfs)
    val_perfs['epoch'] = np.arange(max_epochs)
    val_perfs['run_name'] = run_name
    for key, value in options.items():
      val_perfs[key] = value
    return val_perfs

  def fit(self, X, y, test=None, options=None):
    self.model_stats = pd.concat([self._fit(X, y, o, test=test) for o in options], axis=0)

  def predict(self, X, method=None):
    best_model = self.model_stats.sort_values(method).iloc[-1]
    model_path = self.model_dir / best_model['run_name'] / '{}.pkl'.format(best_model['epoch'])
    model = joblib.load(model_path)
    X = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
      return model(X)

class Model(nn.Module):
  def __init__(self, feat_dim, num_labels):
    super().__init__()
    self.lin = nn.Linear(feat_dim, num_labels)

  def forward(self, X):
    return self.lin(X)

def inject_noise(y, gamma):
  y = np.array(y)
  mask = rn.rand(*y.shape) > gamma
  return (y * (1 - mask) + (1 - y) * mask).astype(bool).astype(int)

def get_20newsgroups():
  data, y = fetch_20newsgroups(
    remove=('headers', 'footers', 'quotes'), return_X_y=True, random_state=0, subset='all'
  )
  ohe = OneHotEncoder()
  y = np.array(ohe.fit_transform(y.reshape(-1, 1)).todense())
  train_data, test_data, train_y, test_y = train_test_split(data, y)
  transformer = CountVectorizer(min_df=10)
  train_X = transformer.fit_transform(train_data).todense()
  test_X = transformer.transform(test_data).todense()
  mean, std = train_X.mean(0), train_X.std(0)
  train_X = (train_X - mean) / std
  test_X = (test_X - mean) / std
  return train_X, test_X, train_y, test_y

def get_synthetic(n=50, d=50, num_queries=10):
  test_size = 100
  theta = torch.randn((num_queries, d))
  X = torch.randn((n + test_size, d))
  y = (torch.rand((n + test_size, num_queries)) < torch.sigmoid((X.unsqueeze(1) * theta.unsqueeze(0)).sum(-1))).float()
  return (a.numpy() for a in (X[:n], X[n : n + test_size], y[:n], y[n : n + test_size]))

fetch_lookup = {
  '20newsgroups': get_20newsgroups,
  'synthetic': get_synthetic
}

def main():
  # experiment_name = '20newsgroups'
  experiment_name = 'synthetic'
  train_X, test_X, train_y, test_y = fetch_lookup[experiment_name]()
  clean_train_y = train_y
  losses = ['logloss', 'ranknet', 'sym_logloss', 'sym_ranknet']
  selection_methods = ['loss', 'auc', 'ndcg@10', 'map']
  model_params = [{'loss': loss}
                  for loss, in product(losses, repeat=1)]
  max_epochs = 50
  # weight_decay_options = np.r_[0, np.geomspace(1e-5, 1e-2, 4)]
  weight_decay_options = [0]
  # lr_options = [0.1]
  lr_options = [0.01]
  options = [{'lr': lr, 'weight_decay': wd, 'max_epochs': max_epochs}
             for lr, wd in product(lr_options, weight_decay_options, repeat=1)]

  gamma = 0.9
  train_y = inject_noise(clean_train_y, gamma)

  performance = []
  for p in model_params:
    model = Estimator(**p)
    model.fit(train_X, train_y, test=(test_X, test_y), options=options)
    for method in selection_methods:
      print('params:', p, 'selection method:', method)
      pred_y = model.predict(test_X, method=method)
      row = dict(**p)
      row['method'] = method
      row['ndcg@10'] = ndcg_score(test_y, pred_y, k=10)
      row['dcg@10'] = dcg_score(test_y, pred_y, k=10)
      row['map'] = average_precision_score(test_y, pred_y)
      row['auc'] = roc_auc_score(test_y, pred_y)
      performance.append(row)
  df = pd.DataFrame(performance)
  df.to_csv('{}.csv'.format(experiment_name))
  print(df)




if __name__ == "__main__": main()
