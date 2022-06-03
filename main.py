import numpy as np
import scipy as sp
from scipy import stats
from scipy.stats import multivariate_normal as normal
import matplotlib.pyplot as plt
import seaborn as sns
import math
from DGMM import DGMM

# l11 = stats.multivariate_normal([0, 0])
# l12 = stats.multivariate_normal([5, 5])
# l21 = stats.multivariate_normal([1, 1])
#
# x1 = np.concatenate([l11.rvs(1000), l12.rvs(500)])
# x2 = l21.rvs(1500) + (np.array([[1, 1], [0, 1]]) @ x1.T).T
#
# plt.figure(figsize=(5, 5))
# plt.plot(x1[:, 0], x1[:, 1], 'b.')
# plt.plot(x2[:, 0], x2[:, 1], 'r.')

l11 = normal([0, 0])
l12 = normal([5, 5])

data = np.concatenate([l11.rvs(100), l12.rvs(100)])

dgmm = DGMM([2, 3, 3], [2, 2, 2], init='random')
clust = dgmm.fit(data, 100)

plt.show()

if __name__ == "__main__":
    pass

