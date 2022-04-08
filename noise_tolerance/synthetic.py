import numpy as np
import pandas as pd
import numpy.random as rn
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score, roc_auc_score, dcg_score, log_loss
from scipy.special import expit

def apply_noise(relevance, gamma):
  mask_0 = rn.rand(*relevance.shape) < gamma[0]
  mask_1 = rn.rand(*relevance.shape) < gamma[1]
  return (relevance * (1 - mask_1) + (1 - relevance) * mask_0).astype(bool).astype(int)

def main():
  n_sim = 1
  n = []
  nn = []
  a = []
  na = []
  for sim_num in range(n_sim):
    n_queries = 100
    n_items = 10000
    n_rankings = 100
    prevalence = [0.0001] * 99 + [0.999] * 1
    # prevalence = np.linspace(0.05, 0.9, n_queries)
    # prevalence = np.linspace(0.1, 0.1, n_queries)
    gamma = np.array([0.1, 0.1])
    relevance = np.zeros((n_queries, n_items))
    for i, p in enumerate(prevalence):
      relevance[i, :int(n_items * p)] = 1
    noisy_relevance = apply_noise(relevance, gamma)
    # scores = [[np.arange(n_items, 0, -1) * (2 * (rn.rand(n_items) < (i / n_rankings)**(ranker_num/n_rankings)) - 1)
    #            for ranker_num, i in enumerate(rn.permutation(range(n_rankings)))]
    #           for _ in range(n_queries)]
    # scores = [[(2*relevance[query_num] - 1) * (2 * (rn.rand(n_items) < (i / n_rankings)**(ranker_num/n_rankings)) - 1)
    #            if ranker_num < n_rankings - 1
    #            # else np.zeros(n_items)
    #            else rn.permutation(range(n_items))
    #            for ranker_num, i in enumerate(rn.permutation(range(n_rankings)))]
    #           for query_num in range(n_queries)]
    scores = [[-np.arange(n_items), np.arange(n_items), rn.permutation(np.arange(n_items))]
              if query_num < 99
              else [np.arange(n_items), -np.arange(n_items), rn.permutation(np.arange(n_items))]
              for query_num in range(n_queries)]
    ndcgs = np.mean([np.array([ndcg_score([r], [score], k=100000) for score in scores]).T
                     for r, scores in zip(relevance, scores)], 0)
    noisy_ndcgs = np.mean([np.array([ndcg_score([r], [score], k=100000) for score in scores]).T
                           for r, scores in zip(noisy_relevance, scores)], 0)
    # ndcgs = np.mean([np.array([log_loss(r, expit(10*score), eps=0) for score in scores]).T
    #                  for r, scores in zip(relevance, scores)], 0)
    # noisy_ndcgs = np.mean([np.array([log_loss(r, expit(10*score), eps=0) for score in scores]).T
    #                        for r, scores in zip(noisy_relevance, scores)], 0)
    aucs = np.mean([np.array([roc_auc_score(r, score) for score in scores]).T
                    for r, scores in zip(relevance, scores)], 0)
    noisy_aucs = np.mean([np.array([roc_auc_score(r, score) for score in scores]).T
                          for r, scores in zip(relevance, scores)], 0)
    n.append(ndcgs)
    nn.append(noisy_ndcgs)
    a.append(aucs)
    na.append(noisy_aucs)

  plt.figure()
  plt.scatter(np.mean(a, 0), np.mean(na, 0))
  plt.figure()
  plt.scatter(np.mean(n, 0), np.mean(nn, 0))

if __name__ == "__main__": main()
