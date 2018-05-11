import numpy as np

class Banana:
    # Parameters from https://github.com/chi-feng/mcmc-demo/blob/19f3fc6a3da0c8afd6ef22af9df1c1b269b34fc4/main/MCMC.js#L29
    precision = np.linalg.inv(np.array([[1,0.5],[0.5,1]]))
    mean = np.array([0.,4.])
    
    @staticmethod
    def _gaussian_log_density(x):
        x = x - Banana.mean
        return 0.5 * np.dot(x, np.dot(Banana.precision, x))

    @staticmethod
    def _gaussian_grad_log_density(x):
        return np.dot(Banana.precision, x - Banana.mean)

    @staticmethod
    def potential(x):
        a, b = 2, 0.2
        y = np.zeros(2)
        y[0] = x[0] / a;
        y[1] = x[1] * a + a * b * (x[0] * x[0] + a * a);
        return Banana._gaussian_log_density(y)

    @staticmethod
    def grad(x):
        a, b = 2, 0.2
        y = np.array([x[0] / a,
                      x[1] * a + a * b * (x[0] * x[0] + a * a)])
        grad = Banana._gaussian_grad_log_density(x)
        return np.array([grad[0] / a + grad[1] * a * b * 2 * x[0],
                         grad[1] * a])


class AlmostDegenerateGaussian:
    precision = np.linalg.inv(np.array([[1.0,1.0 - 1e-4],[1.0 - 1e-4,1.0]]))
    
    @staticmethod
    def potential(x):
        return 0.5 * np.dot(x, np.dot(AlmostDegenerateGaussian.precision, x))

    @staticmethod
    def grad(x):
        return np.dot(AlmostDegenerateGaussian.precision, x)


