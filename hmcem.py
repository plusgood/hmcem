import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def hmcem(x0, potential, grad, eps, L, N, **kwargs):
    M = np.array([[1.0, 0.0], [0.0, 1.0]])
    Mi = np.linalg.inv(M)
    q = x0.copy()

    S_count = kwargs.get('init_S_count', 1000)
    S_I = kwargs.get('S_I', 10)
    nu = kwargs.get('nu', 0.01)
    alpha = kwargs.get('alpha', 1.0)
    z = scipy.stats.norm.ppf((1-alpha/2.0) + (1-alpha/2.0)/2.0)
    d = kwargs.get('d', 2.0)
    samples = []
    num_m_steps = 0

    old_mean = None
    old_std = None

    BURN_IN = 5000
    
    for t in xrange(-BURN_IN, N):
        if t == 0:
            print 'Burn in complete'

        p = np.random.multivariate_normal(mean=np.zeros_like(q), cov=M)
        old_q = q.copy()
        
        old_energy = (potential(q) + 0.5*np.dot(p, np.dot(Mi, p)))

        p -= 0.5 * eps * grad(q)
        for _ in xrange(L-1):
            q += eps * np.dot(Mi, p)
            p -= eps * grad(q)
        q += eps * np.dot(Mi, p)
        p -= 0.5 * eps * grad(q)

        new_energy = (potential(q) + 0.5*np.dot(p, np.dot(Mi, p)))

        if np.random.random() > np.exp(old_energy - new_energy):
            q = old_q

        if t < 0:       # Still in burn-in
            continue

        yield q

        samples.append((p.copy(), q.copy()))

        if len(samples) < S_count:
            continue
        assert len(samples) == S_count

        print 'At time', t, 'performing M step', num_m_steps

        M_est = np.cov([psamp for psamp, _ in samples], rowvar=False)
        print 'M_est', M_est
        Mi_est = np.linalg.inv(M_est)

        kappa = 0.99 / (1.0 + num_m_steps) ** 0.51
        Mi = (1 - kappa) * Mi + kappa * Mi_est
        M = np.linalg.inv(Mi)
        print 'kappa', kappa
        print 'new M!'
        print M
        
        num_m_steps += 1

        subsamples = []
        subsamp_index = 0
        while subsamp_index < S_count:
            subsamples.append(samples[subsamp_index])
            subsamp_index += 1 + np.random.poisson(nu * len(subsamples) ** d)

        print 'Picked', len(subsamples), 'subsamples'

        test_funcs = []
        for psamp, qsamp in subsamples:
            test_funcs.append(np.concatenate((np.dot(Mi, psamp), grad(qsamp))))

        mean = np.mean(test_funcs, axis=0)
        std = np.std(test_funcs, axis=0)
        del test_funcs

        if old_mean is not None:
            """print 'mean', mean
            print 'old_mean', old_mean
            print 'mean - old_mean', mean - old_mean
            print 'z * old_std', z * old_std
            print 'bool', (np.abs(mean - old_mean) < z * old_std)"""

            if (np.abs(mean - old_mean) < z * old_std).any():
                print 'Updating S_count from', S_count, 'to', S_count + S_count // S_I
                S_count = S_count + S_count // S_I
            else:
                print "Didn't update"

        old_mean = mean
        old_std = std

        samples = []
