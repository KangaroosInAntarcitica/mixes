from scipy.stats import multivariate_normal as normal
import numpy as np
from dgmm import Evaluator, GMM, GMN, SkewGMM
import matplotlib.pyplot as plt
import seaborn as sns


def manual_distribution_experiment(distr='gauss', n_samples=2000, levels=5):
    if distr == 'gauss':
        n = normal(np.zeros(2), np.eye(2))
        data = n.rvs(200)
    elif distr == 'quad-gauss':
        n = normal(np.zeros(2), np.eye(2))
        data = n.rvs(n_samples)
        norms = np.apply_along_axis(np.linalg.norm, 1, data).reshape([-1, 1])
        data = data / norms * norms ** 2
        xlim = [-10, 10]
        ylim = [-10, 10]
    elif distr == 'line':
        data = np.linspace(0, 10, n_samples).reshape([-1, 1]) * \
               np.array([1, 1]).reshape([1, -1]) + \
               normal.rvs(np.zeros(2), np.eye(2) / 10, n_samples)
        xlim = ylim = [-1, 11]
    elif distr == 'gauss-2':
        cov = np.array([[10, 0], [0, 1]])
        rot = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4)],
                        [np.sin(np.pi / 4), np.cos(np.pi / 4)]])
        cov = rot @ cov @ rot.T
        data = normal.rvs(np.zeros(2), cov, n_samples)
        xlim = ylim = [-5, 5]
    elif distr == 'skew':
        # shape = np.array([20, 20])
        # cov = np.eye(2)
        # mean = np.zeros(2)
        #
        # # Source:
        # # https://gregorygundersen.com/blog/2020/12/29/multivariate-skew-normal/
        # delta = (1 / np.sqrt(1 + shape @ cov @ shape)) * cov @ shape
        # cov_star = np.block([[np.ones(1), delta],
        #                      [delta[:, None], cov]])
        # mean_star = np.concatenate([[0], mean])
        # v = normal(mean_star, cov_star).rvs(n_samples)
        # data = np.sign(v[:, :1]) * v[:, 1:]

        psi = np.zeros(2)
        sigma = np.eye(2)
        lambd = np.eye(2) * 10

        data = normal.rvs(mean=psi, cov=sigma, size=n_samples) + \
               (lambd @ np.abs(normal.rvs(cov=np.eye(2), size=n_samples).T)).T
        xlim = [-10, 30]
        ylim = [-10, 30]
    else:
        raise ValueError("Distr unsupported")

    evaluator = Evaluator(data)

    fig, ax = plt.subplots(1, 4)

    # Sampled data distribution
    plt.sca(ax[0])
    plt.title("True distribution")
    sns.kdeplot(x=data[:, 0], y=data[:, 1], cmap="rocket", shade=True,
                bw_adjust=2, thresh=-0.01, levels=levels)
    # plt.hist2d(data[:, 0], data[:, 1], bins=30)
    plt.gca().set_aspect('equal')
    plt.xlim(xlim); plt.ylim(ylim)

    # Gaussian Mixture Model
    alg = GMM(1, init='random', num_iter=30,
              evaluator=evaluator, update_rate=1,
              # use_annealing=True, annealing_start_v=0.1,
              plot_predictions=False, plot_wait_for_input=True)
    # alg = GaussianMixture(1)
    alg.fit(data)
    values, dists = alg.random_sample(n_samples)
    # values = normal.rvs(mean=alg.means_[0], cov=alg.covariances_[0], size=n_samples)

    plt.sca(ax[1])
    plt.title("GMM")
    sns.kdeplot(x=values[:, 0], y=values[:, 1], cmap="rocket", shade=True,
                bw_adjust=2, thresh=-0.01, levels=levels)
    plt.gca().set_aspect('equal')
    plt.xlim(xlim); plt.ylim(ylim)

    # Gaussian Mixture Network
    alg = GMN([1, 5], [2, 2], init='random', num_iter=30,
              evaluator=evaluator, update_rate=1, stopping_thresh=0,
              # use_annealing=True, annealing_start_v=0.1,
              plot_predictions=False, plot_wait_for_input=True)
    alg.fit(data)
    values, dists = alg.random_sample(n_samples)

    plt.sca(ax[3])
    plt.title("GMN")
    sns.kdeplot(x=values[:, 0], y=values[:, 1], cmap="rocket", shade=True,
                bw_adjust=2, thresh=-0.01, levels=levels)
    plt.gca().set_aspect('equal')
    plt.xlim(xlim); plt.ylim(ylim)

    # Skew Gaussian Mixture Model
    # alg = SkewGMM(1, num_iter=30, evaluator=evaluator,
    #               update_rate=1,
    #               # use_annealing=True, annealing_start_v=0.1,
    #               plot_predictions=False, plot_wait_for_input=True)
    # alg.fit(data)
    # values, dists = alg.random_sample(n_samples)
    #
    # plt.sca(ax[2])
    # plt.title("Skew GMM")
    # sns.kdeplot(x=values[:, 0], y=values[:, 1], cmap="rocket", shade=True,
    #             bw_adjust=2, thresh=-0.01, levels=levels)
    # plt.gca().set_aspect('equal')
    # plt.xlim(xlim); plt.ylim(ylim)

    # values, dists = alg.random_sample(500)
    # plt.hist2d(values[:, 0], values[:, 1], bins=30)
    # plt.gca().set_aspect('equal')

    plt.show()


if __name__ == "__main__":
    manual_distribution_experiment(distr='skew', n_samples=5000)
    # manual_distribution_experiment(distr='gauss-2', n_samples=500)

