from itertools import count
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ortho_group

class L1SGM():
  def __init__(self, num_rows, dimension, noise_std=1):
    self.num_rows = num_rows
    self.dimension = dimension
    self.A = ortho_group.rvs(num_rows)[:num_rows, :dimension]
    self.x_0 = np.ones(dimension)
    self.y = self.pred(self.x_0, self.A) + np.random.normal(scale=noise_std, size=num_rows)

  def pred(self, x, A):
    return np.matmul(A, x)

  def objective(self, x):
    return np.linalg.norm(self.y - self.pred(x, self.A), ord=1) / len(self.y)

  def subgrad(self, x, y_i, a_i):
    arg = (y_i-np.matmul(a_i, x))
    return -np.matmul(a_i.T, np.sign(arg)) / len(self.y)

  def f(self, x, y_i, a_i):
    return np.linalg.norm(y_i - self.pred(x, a_i), ord=1)

  def minimize(self, init, step_size_sched, step, n_iters=None, stop_error=None, max_iters=None):
    assert (n_iters is None) or (stop_error is None)
    assert not ((n_iters is not None) and (max_iters is not None))
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
      if step in ['sgm', 'gd']: x = x - step_size * g
      elif step == 'trunc': x = x - g * min(step_size, f / np.sum(g ** 2))
      objs.append(self.objective(x))
      xs.append(x)
      if ((stop_error is not None) and (objs[-1] <= stop_error)) or ((max_iters is not None) and (t == max_iters)): break
    return xs, objs

def get_sched(step, init):
  def get_gd_sched(init):
    for i in count(start=1): yield init
  def get_sgm_sched(init):
    for i in count(start=1): yield init / i
  def get_trunc_sched(init):
    for i in count(start=1): yield init * np.power(i, -1/2 - 1e-3)
  if step == 'gd': return get_gd_sched(init)
  elif step == 'sgm': return get_sgm_sched(init)
  elif step == 'trunc': return get_trunc_sched(init)

def main():
  stop_error = 0.001
  max_iters = 1e4
  step_size_inits = 2 ** np.linspace(np.log2(10**2), np.log2(10**6), 10)
  # step_size_inits = 2 ** np.linspace(-4,5, 10) # For GD
  dimension = 40
  num_rows = 1000
  x_init = np.random.normal(size=dimension)

  l1_sgm = L1SGM(num_rows, dimension, noise_std=0)

  for step in ['sgm', 'trunc']:
  # for step in ['gd']:
    plt.figure()
    print(step)
    for init in step_size_inits:
      xs, objs = l1_sgm.minimize(init=x_init,
                                 step_size_sched=get_sched(step=step, init=init),
                                 stop_error=stop_error,
                                 max_iters=max_iters,
                                 step=step)
      print(objs[-1], init)
      plt.plot(np.clip(objs, None, 1e3), label=init)
    plt.legend()
    plt.yscale('log')
    # plt.ylim(10e-3, 10e-2)
    plt.xlabel('Iteration number')
    plt.ylabel('Objective')
    plt.title('Robust Regression ({})'.format(step.upper()))
  plt.show()


if __name__ == "__main__": main()
