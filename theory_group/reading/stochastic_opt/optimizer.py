from abc import ABC, abstractmethod
from itertools import count
import numpy as np
from scipy.stats import ortho_group

class Optimizer(ABC):
  def __init__(self, num_rows, dimension):
    self.num_rows = num_rows
    self.dimension = dimension
    self.A = None
    self.x_0 = None
    self.y = None

  @abstractmethod
  def f(self, x, y_i, a_i): pass

  @abstractmethod
  def objective(self, x): pass

  @abstractmethod
  def subgrad(self, x, y_i, a_i): pass

  def minimize(self,
               init,
               step_size_sched,
               step,
               n_iters=None,
               stop_error=None,
               max_iters=None,
               clip_gamma=None,
               max_norm=None,
               stop_condition=None):
    objs = []
    xs = []
    x = init
    objs.append(self.objective(x))
    xs.append(x)
    counter = range(n_iters) if n_iters is not None else count()
    for t, step_size in zip(counter, step_size_sched):
      i = np.random.choice(np.arange(self.num_rows))
      if step == 'gd':
        y_i, a_i = self.y[:], self.A[:]
      else:
        y_i, a_i = self.y[i][np.newaxis], self.A[i][np.newaxis]
      g = self.subgrad(x, y_i, a_i)
      f = self.f(x, y_i, a_i)
      if step == 'sgm': x = x - step_size * g
      elif step == 'gd': x = x - step_size * g / len(self.y)
      elif step == 'clip': x = x - g * step_size * min(1, clip_gamma / np.linalg.norm(g))
      elif step == 'trunc': x = x - g * min(step_size, f / np.sum(g ** 2))
      if max_norm is not None: x = x / np.linalg.norm(x) * min(max_norm, np.linalg.norm(x))
      objs.append(self.objective(x))
      xs.append(x)
      if ((stop_error is not None) and (objs[-1] <= stop_error)) or ((max_iters is not None) and (t == max_iters)): break
      if stop_condition is not None:
        if stop_condition(x): break
    return xs, objs

def get_sched(step, init):
  def get_gd_sched(init):
    for i in count(start=1): yield init
  def get_sgm_sched(init):
    for i in count(start=1): yield init / i
  def get_trunc_sched(init):
    for i in count(start=1): yield init * np.power(i, -1/2 - 1e-3)
  if step == 'gd': return get_gd_sched(init)
  elif step in ['clip', 'sgm']: return get_sgm_sched(init)
  elif step == 'trunc': return get_trunc_sched(init)
