from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ortho_group
from scipy.special import expit
import sys
try:
  from optimizer import Optimizer, get_sched
except ImportError:
  sys.path.append('./')
  from optimizer import Optimizer, get_sched

class L2BinaryClassification(Optimizer):
  def __init__(self, num_rows, dimension):
    super().__init__(num_rows, dimension)
    self.A = np.random.normal(size=(num_rows, dimension))
    self.x_0 = np.ones(dimension)
    self.y = np.float64(np.random.rand(num_rows) < self.pred(self.x_0, self.A))

  def pred(self, x, A):
    return expit(np.matmul(A, x))

  def objective(self, x):
    return np.linalg.norm(self.y - self.pred(x, self.A), ord=2)**2 / len(self.y)

  def subgrad(self, x, y_i, a_i):
    pred = self.pred(x, a_i)
    return - 2 * np.matmul(a_i.T, (y_i - pred) * pred * (1-pred))

  def f(self, x, y_i, a_i):
    return np.linalg.norm(y_i - self.pred(x, a_i), ord=2)**2

def main():
  stop_error = 0.1
  max_iters = 1e4
  # step_size_inits = 2 ** np.linspace(np.log2(10**-2), np.log2(10**1), 10)
  # step_size_inits = 2 ** np.linspace(np.log2(10**0), np.log2(10**3), 20)
  step_size_inits = 2 ** np.linspace(np.log2(10**0), np.log2(10**5), 10)
  # step_size_inits = 2 ** np.linspace(-15,-10, 10) # For GD
  dimension = 40
  num_rows = 1000
  x_init = np.random.normal(size=dimension)

  l2_class = L2BinaryClassification(num_rows, dimension)

  clip_gamma = 0.25
  f_x_star = l2_class.objective(l2_class.x_0)
  print('best_loss: {}'.format(f_x_star))
  max_norms = [1, 3, 6, 8, 10, None]
  fig, axes = plt.subplots(3, 2, sharex='all', sharey='all')
  axes = axes.reshape(-1)
  for max_norm, ax in zip(max_norms, axes):
    steps_required = defaultdict(dict)
    for step in ['clip', 'sgm', 'trunc']:
      print(step)
      for init in step_size_inits:
        xs, objs = l2_class.minimize(init=x_init,
                                     step_size_sched=get_sched(step=step, init=init),
                                     stop_error=stop_error,
                                     max_iters=max_iters,
                                     step=step,
                                     clip_gamma=clip_gamma,
                                     max_norm=max_norm)
        steps_required[step][init] = len(xs)
        print(np.linalg.norm(xs[-1] - l2_class.x_0)**2, objs[-1], init)
    steps_required = pd.DataFrame(steps_required)
    steps_required.plot(ax=ax)
    ax.set_title('\mathcal X = B_2(0, {})'.format(max_norm) if max_norm is not None else '\mathcal X = \mathbb R^d')
  fig.add_subplot(111, frameon=False)
  # hide tick and tick label of the big axis
  plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
  plt.xlabel('Learning rate')
  plt.ylabel('Iterations until loss = {}'.format(stop_error))
  plt.xscale('log')
  plt.show()


if __name__ == "__main__": main()
