from itertools import count
import numpy as np
import matplotlib.pyplot as plt

class L1SGM():
  def __init__(self, num_rows, dimension, noise_std=1):
    self.num_rows = num_rows
    self.dimension = dimension
    self.A = np.random.normal(loc=10, size=(num_rows, dimension))
    # self.x_0 = np.random.normal(size=dimension)
    self.x_0 = np.ones(dimension)
    self.y = self.pred(self.x_0, self.A) + np.random.normal(scale=noise_std, size=num_rows)

  def pred(self, x, A):
    return np.matmul(A, x)

  def objective(self, x):
    return np.linalg.norm(self.y - self.pred(x, self.A), ord=1) / len(self.y)

  def subgrad(self, x, y_i, a_i):
    arg = (y_i-np.matmul(a_i, x))
    return -np.matmul(a_i.T, np.where(arg > 0, arg, -arg)) / len(self.y)

  def f(self, x, y_i, a_i):
    return np.linalg.norm(y_i - self.pred(x, a_i), ord=1)

  def minimize(self, init, step_size_sched, n_iters, step):
    objs = []
    xs = []
    x = init
    objs.append(self.objective(x))
    xs.append(x.item())
    for t, step_size in zip(range(n_iters), step_size_sched):
      i = np.random.choice(np.arange(self.num_rows))
      if step == 'gd':
        y_i, a_i = self.y[:], self.A[:]
      else:
        y_i, a_i = self.y[i][np.newaxis], self.A[i][np.newaxis]
      g = self.subgrad(x, y_i, a_i).mean(0)
      f = self.f(x, y_i, a_i)
      if step in ['sgm', 'gd']: x = x - step_size * g
      elif step == 'trunc': x = x - g * min(step_size, f / np.sum(g ** 2))
      objs.append(self.objective(x))
      xs.append(x.item())
    return xs, objs

def get_gd_sched(init):
  for i in count(start=1): yield init

def get_sgd_sched(init):
  for i in count(start=1): yield init / i

def get_trunc_sched(init):
  for i in count(start=1): yield init / i

def main():
  dimension = 1
  num_rows = 1000
  x_init = np.ones(dimension) * 0.9
  l1_sgm = L1SGM(num_rows, dimension, noise_std=0)
  for init in [0.001]:
    xs, objs = l1_sgm.minimize(init=x_init,
                               step_size_sched=get_gd_sched(init=init),
                               n_iters=100,
                               step='gd')
  print(xs[-1], objs[-1])
  plt.plot(objs, label=init)
  plt.legend()
  plt.show()

  dimension = 1
  num_rows = 1000
  # x_init = np.random.normal(loc=0.9, scale=0.001, size=dimension)
  x_init = np.ones(dimension) * 0.9
  l1_sgm = L1SGM(num_rows, dimension, noise_std=0)
  for init in [1, 5]:
    xs, objs = l1_sgm.minimize(init=x_init,
                               step_size_sched=get_sgd_sched(init=init),
                               n_iters=10000,
                               step='sgm')
    print(xs[-1], objs[-1])
    plt.plot(objs, label=init)
  plt.legend()
  plt.show()

  dimension = 1
  num_rows = 10000
  x_init = np.ones(dimension)*0.9
  l1_sgm = L1SGM(num_rows, dimension, noise_std=0)
  for init in [1]:
    xs, objs = l1_sgm.minimize(init=x_init,
                               step_size_sched=get_trunc_sched(init=init),
                               n_iters=10000,
                               step='trunc')
  print(xs[-1], objs[-1])

  plt.plot(objs, label=init)
  plt.legend()
  plt.show()

if __name__ == "__main__": main()
