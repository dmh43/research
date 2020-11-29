import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ortho_group
import sys
try:
  from optimizer import Optimizer, get_sched
except ImportError:
  sys.path.append('./')
  from optimizer import Optimizer, get_sched

class L1(Optimizer):
  def __init__(self, dimension):
    super().__init__(1, dimension)
    self.A = np.ones((1, dimension))
    self.x_0 = np.zeros(dimension)
    self.y = np.zeros(1)

  def pred(self, x, A):
    return np.matmul(A, x)

  def objective(self, x):
    return np.linalg.norm(self.y - self.pred(x, self.A), ord=1)

  def subgrad(self, x, y_i, a_i):
    arg = (y_i-np.matmul(a_i, x))
    return -np.matmul(a_i.T, np.sign(arg))

  def f(self, x, y_i, a_i):
    return np.linalg.norm(y_i - self.pred(x, a_i), ord=1)

def main():
  stop_error = 0.001
  max_iters = 1e2
  init = 1
  dimension = 40
  x_init = np.random.normal(loc=20, size=dimension)

  l1 = L1(dimension)

  clip_gamma = 3
  for step in ['clip', 'sgm', 'trunc']:
    print(step)
    xs, objs = l1.minimize(init=x_init,
                           step_size_sched=get_sched(step=step, init=init),
                           stop_error=stop_error,
                           max_iters=max_iters,
                           step=step,
                           clip_gamma=clip_gamma)
    print(objs[-1], init)
    plt.plot([np.linalg.norm(x - l1.x_0)**2 for x in xs], label=step)
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Iteration number')
    plt.ylabel('dist^2')
    plt.title('Minimize |x|_1')
  plt.show()


if __name__ == "__main__": main()
