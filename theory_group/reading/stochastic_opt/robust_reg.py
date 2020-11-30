import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ortho_group
import sys
try:
  from optimizer import Optimizer, get_sched
except ImportError:
  sys.path.append('./')
  from optimizer import Optimizer, get_sched

class L1Regression(Optimizer):
  def __init__(self, num_rows, dimension, noise_std):
    super().__init__(num_rows, dimension)
    self.A = ortho_group.rvs(num_rows)[:num_rows, :dimension]
    self.x_0 = np.ones(dimension)
    self.y = self.pred(self.x_0, self.A) + np.random.normal(scale=noise_std, size=num_rows)

  def pred(self, x, A):
    return np.matmul(A, x)

  def objective(self, x):
    return np.linalg.norm(self.y - self.pred(x, self.A), ord=1) / len(self.y)

  def subgrad(self, x, y_i, a_i):
    arg = (y_i-np.matmul(a_i, x))
    return -np.matmul(a_i.T, np.sign(arg))

  def f(self, x, y_i, a_i):
    return np.linalg.norm(y_i - self.pred(x, a_i), ord=1)

def main():
  stop_error = 0.001
  max_iters = 1e4
  step_size_inits = 2 ** np.linspace(np.log2(10**-3), np.log2(10**6), 10)
  # step_size_inits = 2 ** np.linspace(-12,-8, 10) # For GD
  dimension = 40
  num_rows = 1000
  x_init = np.random.normal(size=dimension)

  l1_reg = L1Regression(num_rows, dimension, noise_std=0)

  clip_gamma = 0.25
  for step in ['clip', 'sgm', 'trunc']:
  # for step in ['gd']:
    plt.figure()
    print(step)
    for init in step_size_inits:
      xs, objs = l1_reg.minimize(init=x_init,
                                 step_size_sched=get_sched(step=step, init=init),
                                 stop_error=stop_error,
                                 max_iters=max_iters,
                                 step=step,
                                 clip_gamma=clip_gamma)
      print(objs[-1], init)
      # plt.plot(np.clip(objs, None, 1e3), label=init)
      plt.plot([np.linalg.norm(x - l1_reg.x_0)**2 for x in xs], label=init)
    plt.legend()
    plt.yscale('log')
    # plt.ylim(10e-3, 10e-2)
    plt.xlabel('Iteration number')
    # plt.ylabel('Objective')
    plt.ylabel('dist^2')
    plt.title('Robust Regression ({})'.format(step.upper()))
  plt.show()


if __name__ == "__main__": main()
