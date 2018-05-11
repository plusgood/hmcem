import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from hmc import hmc
from hmcem import hmcem
from distributions import *


samps = []

sample_generator = hmc(np.array([0.0, 0.0]),
                         Banana.potential,
                         Banana.grad,
                         0.01,
                         20,
                         10000)

for i, q in enumerate(sample_generator):
    if i % 100 == 0:
        print i, q[0], q[1]
    samps.append((q[0], q[1]))

a,b  = zip(*samps)

#plt.scatter(a,b)
plt.hist2d(a, b, (20, 20), cmap=plt.cm.jet)
plt.colorbar()
plt.show()

#plt.hist(a, bins=15);plt.show()

#plt.hist(b, bins=15);plt.show()

#Degen:
#eps = 0.01, L = 20, sampsize = 10000, starting point = 1.0,1.0


#Banana:
