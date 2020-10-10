from rpca import rpca
# import cvxpy as cp
import os
from scipy.io import wavfile
import scipy.signal as sig
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def play(audio, sample_rate):
  wavfile.write('tmp.wav', sample_rate, audio.astype(np.int16))
  os.system('afplay {}'.format('tmp.wav'))

def say(text):
  os.system('say {}'.format(text))

def objective(L, S, M, lagrange_lambda, rho, beta):
  const = (L + S - M)
  lagrange_term = (lagrange_lambda * const).sum()
  aug_term = rho / 2 * (const ** 2).sum()
  return la.norm(L, ord='nuc') + beta * la.norm(S, ord=1) + lagrange_term + aug_term

def solve(M, beta):
  L = cp.Variable(shape=M.shape, complex=True)
  S = cp.Variable(shape=M.shape, complex=True)
  objective = cp.Minimize(cp.normNuc(L) + beta * cp.norm1(S))
  constraints = [L + S == M]
  problem = cp.Problem(objective, constraints)
  problem.solve(verbose=True)
  return L.value, S.value

def main():
  sample_rate, data = wavfile.read('./mixture2.wav')
  f, t, Zxx = sig.stft(data, fs=sample_rate)
  beta = 1/np.sqrt(max(Zxx.shape))
  M = Zxx / la.norm(Zxx)

  # L, S = solve(M, beta)
  L, S, _, _ = rpca(np.absolute(M), eps_dual=1e-10, verbose=True, max_iter=100)
  phase = np.exp(1j * np.angle(M))
  L_it, L_ift = sig.istft(L * phase * la.norm(Zxx), fs=sample_rate)
  S_it, S_ift = sig.istft(S * phase * la.norm(Zxx), fs=sample_rate)
  reconstruction = L_ift + S_ift
  it, ift = sig.istft(Zxx, fs=sample_rate)
  print('loss:', ((reconstruction - ift)**2).sum())
  say('reconstruction')
  play(ift, sample_rate)
  say('robust PCA')
  say('low rank')
  play(L_ift, sample_rate)
  say('sparse')
  play(S_ift, sample_rate)

  gain = 2
  Mb = np.absolute(L) > np.absolute(S) * gain
  L_it, L_ift = sig.istft(M * Mb * la.norm(Zxx), fs=sample_rate)
  S_it, S_ift = sig.istft(M * (1-Mb) * la.norm(Zxx), fs=sample_rate)
  say('with masking')
  say('background')
  play(L_ift, sample_rate)
  say('foreground')
  play(S_ift, sample_rate)

if __name__ == "__main__": main()
