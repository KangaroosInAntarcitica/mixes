from sklearn.decomposition import FactorAnalysis
import numpy as np
from scipy.stats import multivariate_normal as normal
from dgmm import AbstractDGMM


# fa = FactorAnalysis(n_components=3, rotation='varimax')
#
# data = np.random.random([1000, 8])
# d1 = fa.fit_transform(data)
# fa.ana
# print("res = ", np.mean(np.abs(data - fa.mean_ - d1 @ fa.components_)))

def exp_2():
    a = normal(np.random.random(3),
               cov=AbstractDGMM.make_spd(np.random.random([3, 3])),
               allow_singular=True)
    lambd = np.random.random([4, 3])
    b = normal(np.random.random(4),
               cov=AbstractDGMM.make_spd(np.random.random([4, 4])),
               allow_singular=True)

    exp_a = 0
    exp_b = 0
    exp_a_b = 0
    n = 1000
    denom = 0

    for i in range(n):
        x = a.rvs()
        mean = (lambd @ x.reshape([-1, 1])).reshape(-1)
        y = b.rvs() + mean

        prob = a.pdf(x) * normal(b.mean + mean, b.cov, allow_singular=True).pdf(y)

        exp_a += x.reshape([-1, 1]) * prob
        exp_b += y.reshape([-1, 1]) * prob
        exp_a_b += (x.reshape([-1, 1]) @ y.reshape([1, -1])) * prob
        denom += prob

    exp_a /= denom
    exp_b /= denom
    exp_a_b /= denom

    print(exp_a @ exp_b.T)
    print(exp_a_b)


if __name__ == "__main__":
    exp_2()
