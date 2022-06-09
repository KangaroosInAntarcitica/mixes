import numpy as np
import scipy as sp
from scipy import stats
from scipy.stats import multivariate_normal as normal
import matplotlib.pyplot as plt
import seaborn as sns
import math
from DGMM import DGMM
from sklearn import metrics

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

l11 = normal([0, 0, 0])
l12 = normal([5, 5, 0])
l13 = normal([-5, 7, 0])

data = np.concatenate([l11.rvs(500), l12.rvs(500), l13.rvs(500)])
labels = np.concatenate([np.ones(500) * 0, np.ones(500) * 1, np.ones(500) * 2])

dgmm = DGMM([3, 4, 2], [3, 2, 2], init='random')
probs = dgmm.fit(data, 10)
clust = np.argmax(probs, axis=1)

print("ARI = ", metrics.adjusted_rand_score(labels, clust))

plt.show()

if __name__ == "__main__":
    pass

