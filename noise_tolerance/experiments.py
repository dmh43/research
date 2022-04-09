from itertools import product, groupby
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import numpy.random as rn
import matplotlib.pyplot as plt

import joblib

from sklearn.datasets import fetch_20newsgroups, load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import ndcg_score, roc_auc_score, average_precision_score, dcg_score
from sklearn.linear_model import LogisticRegressionCV

import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm

def qg_split(coll, qgs):
  return np.array_split(coll, np.cumsum(qgs)[:-1])

def wrap_qgs(metric_fn):
  def qg_metric(a, b, qgs=None, **kwargs):
    a = qg_split(a, qgs)
    b = qg_split(b, qgs)
    return np.mean([metric_fn(ai, bi, **kwargs) for ai, bi in zip(a, b) if np.std(ai) != 0])
  return qg_metric

def wrap_dcg(dcg_fn):
  def better_dcg(a, b, k=None):
    if len(a.shape) == 1:
      return dcg_fn([a], [b], k=k)
    else:
      return dcg_fn(a, b, k=k)
  return better_dcg

def get_selection_lookup(loss_fn):
  return {
    'auc': roc_auc_score,
    'dcg@10': lambda a, b: wrap_dcg(dcg_score)(a, b, k=10),
    'ndcg@10': lambda a, b: wrap_dcg(ndcg_score)(a, b, k=10),
    'map': average_precision_score,
    'loss': lambda a, b: -loss_fn(torch.tensor(b), torch.tensor(a)).item()
  }

class QBCEWithLogitsLoss():
  def __call__(self, out, y, qgs=None):
    return nn.BCEWithLogitsLoss()(out, y)

class SymBCEWithLogitsLoss():
  def __call__(self, out, y, qgs=None):
    return -(torch.sigmoid(out) * y + (1 - torch.sigmoid(out)) * (1 - y)).mean()

class RanknetLoss():
  def __call__(self, out, y):
    losses = []
    for i in range(y.shape[1]):
      out_rel = out[y[:, i] == 1][:, i]
      out_irrel = out[y[:, i] == 0][:, i]
      out_diff = (out_rel.unsqueeze(0) - out_irrel.unsqueeze(1)).reshape(-1)
      losses.append(F.binary_cross_entropy_with_logits(out_diff, torch.ones_like(out_diff)))
    return torch.mean(torch.stack(losses))

class SymRanknetLoss():
  def __call__(self, out, y):
    losses = []
    for i in range(y.shape[1]):
      out_rel = out[y[:, i] == 1][:, i]
      out_irrel = out[y[:, i] == 0][:, i]
      out_diff = (out_rel.unsqueeze(0) - out_irrel.unsqueeze(1)).reshape(-1)
      losses.append(SymBCEWithLogitsLoss()(out_diff, torch.ones_like(out_diff)))
    return torch.mean(torch.stack(losses))

class QSymRanknetLoss():
  def __call__(self, out, y, qgs=None):
    losses = []
    for q_out, q_y in zip(qg_split(out, qgs), qg_split(y, qgs)):
      out_rel = q_out[q_y == 1]
      out_irrel = q_out[q_y == 0]
      out_diff = (out_rel.unsqueeze(0) - out_irrel.unsqueeze(1)).reshape(-1)
      losses.append(SymBCEWithLogitsLoss()(out_diff, torch.ones_like(out_diff)))
    return torch.mean(torch.stack(losses))

class QRanknetLoss():
  def __call__(self, out, y, qgs=None):
    losses = []
    for q_out, q_y in zip(qg_split(out, qgs), qg_split(y, qgs)):
      out_rel = q_out[q_y == 1]
      out_irrel = q_out[q_y == 0]
      out_diff = (out_rel.unsqueeze(0) - out_irrel.unsqueeze(1)).reshape(-1)
      losses.append(F.binary_cross_entropy_with_logits(out_diff, torch.ones_like(out_diff)))
    return torch.mean(torch.stack(losses))

class Estimator():
  def __init__(self, *, loss=None):
    self.loss = loss
    self.model_dir = Path('models')
    self.model_stats = None
    self.train_losses = []
    self.test_losses = []

  def _fit(self, X, y, options, test=None): pass

  def fit(self, X, y, qgs, test=None, options=None):
    self.model_stats = pd.concat([self._fit(X, y, qgs, o, test=test) for o in options], axis=0)

  def predict(self, X, method=None):
    best_model = self.model_stats.sort_values(method).iloc[-1]
    model_path = self.model_dir / best_model['run_name'] / '{}.pkl'.format(best_model['epoch'])
    model = joblib.load(model_path)
    X = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
      return model(X)

class QScorerEstimator(Estimator):
  def _fit(self, X, y, qgs, options, test=None):
    model = Model(X.shape[1], 1)
    run_name = str(hash((self.loss, *sorted(options.items(), key=lambda pair: pair[0]))))
    run_dir = self.model_dir / run_name
    run_dir.mkdir(exist_ok=True)
    max_epochs = options['max_epochs']
    weight_decay = options['weight_decay']
    lr = options['lr']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = {'logloss': QBCEWithLogitsLoss,
               'ranknet': QRanknetLoss,
               'sym_logloss': SymBCEWithLogitsLoss,
               'sym_ranknet': QSymRanknetLoss
               }[self.loss]()
    selection_lookup = {k: wrap_qgs(v) for k, v in get_selection_lookup(loss_fn).items()}
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    train_qgs, val_qgs = qgs[:3*len(qgs) // 4], qgs[3*len(qgs) // 4:]
    train_X = X[:sum(train_qgs)]
    train_y = y[:sum(train_qgs)]
    val_X = X[sum(train_qgs):]
    val_y = y[sum(train_qgs):]
    val_perfs = []
    print(options)
    for i in tqdm(range(max_epochs)):
      epoch_losses = []
      for q_X, q_y in zip(qg_split(train_X, train_qgs), qg_split(train_y, train_qgs)):
        optimizer.zero_grad()
        out = model(q_X)
        loss = loss_fn(out, q_y)
        epoch_losses.append(loss.item())
        loss.backward()
        optimizer.step()
      self.train_losses.append(np.mean(epoch_losses))
      joblib.dump(model, run_dir / '{}.pkl'.format(i))
      val_perfs.append({name: fn(val_y.numpy(), model(val_X).detach().numpy(), qgs=val_qgs)
                        for name, fn in selection_lookup.items()})
      if i % 10 == 0:
        print('training loss: {:.4f}, val: {}'.format(
          loss.item(),
          str({k: '{:.4f}'.format(v) for k, v in val_perfs[-1].items()})
        ))
        if test:
          test_X, test_y, test_qgs = test
          test_X = torch.tensor(test_X, dtype=torch.float32)
          test_perf = ['{}: {:.4f}'
                       .format(name, fn(np.array(test_y), model(test_X).detach().numpy(), qgs=test_qgs))
                       for name, fn in selection_lookup.items()]
          self.test_losses.append(loss_fn(model(test_X).detach(), torch.tensor(test_y), qgs=test_qgs))
          print('test perf: {}'.format(' '.join(test_perf)))
    val_perfs = pd.DataFrame(val_perfs)
    val_perfs['epoch'] = np.arange(max_epochs)
    val_perfs['run_name'] = run_name
    for key, value in options.items():
      val_perfs[key] = value
    return val_perfs

class ScorerEstimator(Estimator):
  def _fit(self, X, y, qgs, options, test=None):
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
    selection_lookup = get_selection_lookup(loss_fn)
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
          test_X, test_y, test_qgs = test
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

class Model(nn.Module):
  def __init__(self, feat_dim, num_labels):
    super().__init__()
    self.num_labels = num_labels
    self.lin = nn.Linear(feat_dim, num_labels)

  def forward(self, X):
    out = self.lin(X)
    if self.num_labels == 1:
      return out.reshape(-1)
    else:
      return out

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
  return train_X, test_X, train_y, test_y, None, None

def get_synthetic(n=50, d=50, num_queries=10):
  test_size = 100
  theta = torch.randn((num_queries, d))
  X = torch.randn((n + test_size, d))
  y = (torch.rand((n + test_size, num_queries)) < torch.sigmoid((X.unsqueeze(1) * theta.unsqueeze(0)).sum(-1))).float()
  return [a.numpy() for a in (X[:n], X[n : n + test_size], y[:n], y[n : n + test_size])], None, None

def get_mq_2007(path='./data/mq2007.txt'):
  X, y, row_qids = load_svmlight_file(path, query_id=True)
  qids, qgs = zip(*[(g, len(list(keys))) for g, keys in groupby(row_qids)])
  X = X.todense()
  y = y.clip(0, 1)
  to_drop = {i for i, a in enumerate(qg_split(y, qgs)) if (max(a) == 0) or (min(a) == 1)}
  X = np.concatenate([q_X for i, q_X in enumerate(qg_split(X, qgs)) if i not in to_drop])
  y = np.concatenate([q_y for i, q_y in enumerate(qg_split(y, qgs)) if i not in to_drop])
  row_qids = np.concatenate([q_id for i, q_id in enumerate(qg_split(row_qids, qgs)) if i not in to_drop])
  qids, qgs = zip(*[(g, len(list(keys))) for g, keys in groupby(row_qids)])
  train_qids, test_qids = train_test_split(qids)
  train_qids = set(train_qids)
  train_mask = np.array([True if qid in train_qids else False for qid in row_qids])
  train_qgs = [qg for qg, qid in zip(qgs, qids) if qid in train_qids]
  test_qgs = [qg for qg, qid in zip(qgs, qids) if qid not in train_qids]
  X = 10 * X
  return X[train_mask], X[~train_mask], y[train_mask], y[~train_mask], train_qgs, test_qgs

def get_mq_2008(path='./data/mq2008.txt'):
  return get_mq_2007(path=path)

fetch_lookup = {
  '20newsgroups': get_20newsgroups,
  'synthetic': get_synthetic,
  'mq2007': get_mq_2007,
  'mq2008': get_mq_2008
}

estimator_lookup = {
  '20newsgroups': ScorerEstimator,
  'synthetic': ScorerEstimator,
  'mq2007': QScorerEstimator,
  'mq2008': QScorerEstimator
}

def main():
  # experiment_name = '20newsgroups'
  # experiment_name = 'synthetic'
  experiment_name = 'mq2008'
  train_X, test_X, train_y, test_y, train_qgs, test_qgs = fetch_lookup[experiment_name]()
  clean_train_y = train_y

  losses = ['logloss', 'ranknet', 'sym_logloss', 'sym_ranknet']
  # losses = ['sym_logloss']
  selection_methods = ['loss', 'auc', 'ndcg@10', 'dcg@10', 'map']
  model_params = [{'loss': loss}
                  for loss, in product(losses, repeat=1)]
  # max_epochs = 50
  # max_epochs = 300
  max_epochs = 30
  # weight_decay_options = np.r_[0, np.geomspace(1e-5, 1e-2, 4)]
  # weight_decay_options = [0]
  weight_decay_options = [1e-5]
  # lr_options = [0.01]
  lr_options = [0.0001]
  options = [{'lr': lr, 'weight_decay': wd, 'max_epochs': max_epochs, 'exp_name': experiment_name}
             for lr, wd in product(lr_options, weight_decay_options, repeat=1)]

  performance = []
  for gamma in [1, 0.9, 0.8, 0.7, 0.6]:
    train_y = inject_noise(clean_train_y, gamma)
    for p in model_params:
      torch.manual_seed(0)
      model = estimator_lookup[experiment_name](**p)
      model.fit(train_X, train_y, train_qgs, test=(test_X, test_y, test_qgs), options=options)
      for method in selection_methods:
        pred_y = model.predict(test_X, method=method).numpy()
        row = dict(**p)
        row['method'] = method
        row['ndcg@10'] = wrap_qgs(wrap_dcg(ndcg_score))(test_y, pred_y, k=10, qgs=test_qgs)
        row['dcg@10'] = wrap_qgs(wrap_dcg(dcg_score))(test_y, pred_y, k=10, qgs=test_qgs)
        row['map'] = wrap_qgs(average_precision_score)(test_y, pred_y, qgs=test_qgs)
        row['auc'] = wrap_qgs(roc_auc_score)(test_y, pred_y, qgs=test_qgs)
        row['gamma'] = gamma
        performance.append(row)
  df = pd.DataFrame(performance)
  df.to_csv('{}.csv'.format(experiment_name))
  print(df)




if __name__ == "__main__": main()
