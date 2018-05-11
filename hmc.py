import numpy as np
import scipy.stats


def hmc_single_step(x0, potential, grad, eps, L):
    M = np.array([[1.0, 0.0], [0.0, 1.0]])
    Mi = np.linalg.inv(M)

    q = x0.copy()
    p = np.random.multivariate_normal(mean=np.zeros_like(x0), cov=M)
    
    old_energy = (potential(q) + 0.5*np.dot(p, np.dot(Mi, p)))

    p -= 0.5 * eps * grad(q)
    for t in xrange(L-1):
        q += eps * np.dot(Mi, p)
        p -= eps * grad(q)
    q += eps * np.dot(Mi, p)
    p -= 0.5 * eps * grad(q)

    new_energy = (potential(q) + 0.5*np.dot(p, np.dot(Mi, p)))

    if np.random.random() < np.exp(old_energy - new_energy):
        return q
    else:
        return x0

def hmc(x0, potential, grad, eps, L, N):
    x = x0
    for i in xrange(-5000, N):
        x = hmc_single_step(x.copy(), potential, grad, eps, L)
        if i >= 0:
            yield x

