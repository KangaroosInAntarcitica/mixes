from scipy.stats import multivariate_normal as normal
import numpy as np
from mixes import Evaluator, GMM, GMN, SkewGMM
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn import preprocessing


def fit_algorithms_to_distribution(data, n_samples, algorithms, algorithm_names,
                                   xlim, ylim, plot_levels=5, plot_bw_adjust=2,
                                   separate_plots=False):
    evaluator = Evaluator(data)

    scale = 2
    if not separate_plots:
        fig, ax = plt.subplots(1, len(algorithms) + 1,
                               figsize=((len(algorithms) + 1) * scale * 1.2, scale))

    # Sampled data distribution
    if separate_plots:
        plt.figure(figsize=(2, 2), dpi=300)
    else:
        plt.sca(ax[0])
        plt.title("True distribution")
    sns.kdeplot(x=data[:, 0], y=data[:, 1], cmap="rocket", shade=True,
                bw_adjust=plot_bw_adjust, thresh=-0.01, levels=plot_levels)
    # plt.gca().set_aspect('equal')
    plt.xlim(xlim)
    plt.ylim(ylim)

    if separate_plots:
        plt.show()

    result = []
    log_lik = []

    for i in range(len(algorithms)):
        alg = algorithms[i]
        alg.evaluator = evaluator
        name = algorithm_names[i]

        alg.fit(data)
        values, dists = alg.random_sample(n_samples)

        if separate_plots:
            plt.figure(figsize=(2, 2), dpi=300)
        else:
            plt.sca(ax[i + 1])
            plt.title(name)
        sns.kdeplot(x=values[:, 0], y=values[:, 1], cmap="rocket", shade=True,
                    bw_adjust=plot_bw_adjust, thresh=-0.01, levels=plot_levels)
        # plt.hist2d(values[:, 0], values[:, 1], bins=30)
        # plt.gca().set_aspect('equal')
        plt.xlim(xlim)
        plt.ylim(ylim)
        log_lik.append(evaluator.get_result_metric("log_lik"))
        result.append(alg)

        if separate_plots:
            plt.show()

    if not separate_plots:
        plt.show()

    print("\n".join(["%s log-likelihood = \t%12.7f" %
          (algorithm_names[i], log_lik[i]) for i in range(len(algorithms))]))
    return result


def skew_gaussian_distribution_experiment(n_samples, seed=None, separate_plots=False):
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

    if seed is not None:
        np.random.seed(seed)

    psi = np.zeros(2)
    sigma = np.eye(2)
    lambd = np.eye(2) * 10

    data = normal.rvs(mean=psi, cov=sigma, size=n_samples) + \
           (lambd @ np.abs(normal.rvs(cov=np.eye(2), size=n_samples).T)).T

    gmm = GMM(1, init='random', num_iter=100, update_rate=1)
    gmn = GMN([1, 5], [2, 2], init='random', num_iter=100,
              update_rate=1, stopping_thresh=0)
    skew = SkewGMM(1, num_iter=30, update_rate=1)
    algorithms = [gmm, gmm, gmn]
    algorithm_names = ["GMM", "Skew GMM", "GMN"]

    xlim, ylim = [-10, 30], [-10, 30]
    fit_algorithms_to_distribution(
        data, n_samples, algorithms, algorithm_names,
        plot_levels=20,
        xlim=xlim, ylim=ylim,
        separate_plots=separate_plots)

    plt.figure(figsize=(3, 3), dpi=200)
    gmn.plot_distributions(data, different_path_colors=True)
    plt.xlim([0, 20])
    plt.ylim([0, 20])
    plt.title("")
    plt.show()


def old_faithful_distribution_experiment(n_samples=200, seed=None):
    # Source https://gist.github.com/curran/4b59d1046d9e66f2787780ad51a1cd87#file-data-tsv
    # and https://github.com/mwaskom/seaborn-data/blob/master/geyser.csv

    if seed is not None:
        np.random.seed(seed)

    geyser = sns.load_dataset("geyser")
    data = geyser[["waiting", "duration"]].values
    data = preprocessing.scale(data)
    geyser[["waiting", "duration"]] = data

    plt.figure(figsize=(3, 3), dpi=200)
    sns.kdeplot(data=geyser, x="waiting", y="duration", hue="kind")
    plt.show()

    evaluator = Evaluator(data=data)

    gmn = GMN([2, 5, 2], [2, 2, 1], num_iter=200, update_rate=0.1, init='kmeans',
              evaluator=evaluator)
    gmn.fit(data)
    values, dists = gmn.random_sample(n_samples)
    plt.figure(figsize=(3, 3), dpi=200)
    sns.kdeplot(x=values[:,0], y=values[:,1], hue=dists)
    plt.show()

    gmm = GMM(2, init='random', num_iter=100, update_rate=1,
              evaluator=evaluator)
    gmm.fit(data)
    values, dists = gmm.random_sample(n_samples)
    plt.figure(figsize=(3, 3), dpi=200)
    sns.kdeplot(x=values[:, 0], y=values[:, 1], hue=dists)
    plt.show()

    plt.figure(figsize=(3, 3), dpi=200)
    gmn.plot_distributions(data, different_path_colors=False, use_pi=True)
    plt.show()

    # algorithms = [gmm, gmn]
    # algorithm_names = ["GMM", "GMN"]
    # fit_algorithms_to_distribution(
    #     data=data, n_samples=n_samples, algorithms=algorithms,
    #     algorithm_names=algorithm_names, xlim=[0, 6], ylim=[40, 100],
    #     plot_levels=20, separate_plots=True, plot_bw_adjust=0.4)



def fit_distribution(data, n_samples=2000, plot_levels=5, plot_bw_adjust=2):
    # if distr == 'gauss':
    #     n = normal(np.zeros(2), np.eye(2))
    #     data = n.rvs(200)
    # elif distr == 'quad-gauss':
    #     n = normal(np.zeros(2), np.eye(2))
    #     data = n.rvs(n_samples)
    #     norms = np.apply_along_axis(np.linalg.norm, 1, data).reshape([-1, 1])
    #     data = data / norms * norms ** 2
    #     xlim = [-10, 10]
    #     ylim = [-10, 10]
    # elif distr == 'line':
    #     data = np.linspace(0, 10, n_samples).reshape([-1, 1]) * \
    #            np.array([1, 1]).reshape([1, -1]) + \
    #            normal.rvs(np.zeros(2), np.eye(2) / 10, n_samples)
    #     xlim = ylim = [-1, 11]
    # elif distr == 'gauss-2':
    #     cov = np.array([[10, 0], [0, 1]])
    #     rot = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4)],
    #                     [np.sin(np.pi / 4), np.cos(np.pi / 4)]])
    #     cov = rot @ cov @ rot.T
    #     data = normal.rvs(np.zeros(2), cov, n_samples)
    #     xlim = ylim = [-5, 5]
    pass


if __name__ == "__main__":
    skew_gaussian_distribution_experiment(n_samples=2000, separate_plots=False)
    # old_faithful_distribution_experiment(n_samples=2000, seed=10)
