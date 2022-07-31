import numpy as np
from scipy.stats import multivariate_normal as normal
from scipy.stats import skewnorm as skew_normal
import matplotlib.pyplot as plt
from mixes import SamplingDGMM as DGMM, SkewGMM
from mixes import GradientDescentDGMM as GDGMM
from mixes import GMM
from mixes import GMN
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import pandas as pd
from sklearn import preprocessing, datasets, metrics
from mixes import Evaluator


def test_on_data(alg, data, labels, rescale=True):
    if rescale:
        data = preprocessing.scale(data)

    print("Testing on data. Num classes = %d, num rows = %d"
          % (len(np.unique(labels)), len(data)))

    if hasattr(alg, "fit_predict"):
        clust = alg.fit_predict(data)
    else:
        clust = alg.fit(data)

    print()
    print("Silhouette score = ", metrics.silhouette_score(data, labels))
    print("Accuracy = ", Evaluator.accuracy(labels, clust))
    print("ARI = ", metrics.adjusted_rand_score(labels, clust))

    try:
        plt.pause(1e10)
    except Exception as e:
        pass


def load_ecoli():
    data = pd.read_csv('data/ecoli.csv', header=None)
    data, labels = data.values[:,:-1].astype('float'), data.values[:,-1]
    mapping = {v: i for i, v in enumerate(np.unique(labels))}
    labels = np.array([mapping[x] for x in labels])
    return data, labels


def manual_dataset():
    l11 = normal([0, 0, 0])
    l12 = normal([5, 5, 0])
    l13 = normal([-5, 10, 0])
    l14 = normal([-5, 3, 0])

    data = np.concatenate([
        l11.rvs(500),
        l12.rvs(500),
        l13.rvs(500),
        l14.rvs(500)
    ])
    labels = np.concatenate([
        np.ones(500) * 0,
        np.ones(500) * 1,
        np.ones(500) * 2,
        np.ones(500) * 2
    ])

    return data, labels


def manual_dataset_2():
    x = 5
    y = 5
    l11 = normal([-x, 0, 0])
    l12 = normal([-x, y, 0])
    l13 = normal([x, 0, 0])
    l14 = normal([x, y, 0])

    data = np.concatenate([
        l11.rvs(500),
        l12.rvs(500),
        l13.rvs(500),
        l14.rvs(500)
    ])
    labels = np.concatenate([
        np.ones(500) * 0,
        np.ones(500) * 0,
        np.ones(500) * 1,
        np.ones(500) * 1
    ])

    return data, labels


def try_random(algorithm='dgmm', ndims=2, nclust=6, size=1000):
    data = []
    labels = []

    for i in range(nclust):
        cov = GMM.make_spd(np.random.random([ndims, ndims]))
        clust_size = int(size / nclust)
        distrib = normal(np.random.random(ndims), cov)
        data.append(distrib.rvs(clust_size))
        labels.append(np.repeat(i, clust_size))

    data = np.concatenate(data)
    labels = np.concatenate(labels)
    evaluator = Evaluator(data, labels, 'silhouette', 'accuracy', 'ARI')

    if algorithm == 'dgmm':
        alg = DGMM([2], [3], init='kmeans', plot_predictions=1,
                   plot_wait_for_input=True,
                   num_iter=100, num_samples=5000,
                   evaluator=evaluator)

    test_on_data(alg, data, labels, rescale=False)


def try_manual(algorithm='dgmm'):
    data, labels = manual_dataset_2()
    evaluator = Evaluator(data, labels, 'silhouette', 'accuracy', 'ARI')

    if algorithm == 'dgmm':
        alg = DGMM([2, 2], [2, 1], init='kmeans', plot_predictions=10,
                   plot_wait_for_input=False,
                   num_iter=100,
                   evaluator=evaluator)
    elif algorithm == 'gdgmm':
        alg = GDGMM([2, 2], [2, 1], init='kmeans', plot_predictions=10,
                   plot_wait_for_input=False,
                   num_iter=100, step_size=1e-2, num_samples=100,
                   evaluator=evaluator)
    elif algorithm == 'gmm':
        alg = GMM(2, init='random', plot_predictions=1, num_iter=40,
                  plot_wait_for_input=True,
                  evaluator=evaluator)
    elif algorithm == 'kmeans':
        alg = KMeans(3, max_iter=100, n_init=30)

    test_on_data(alg, data, labels, rescale=False)


def try_ecoli(algorithm='dgmm'):
    data, labels = load_ecoli()
    # alg = KMeans(3, max_iter=100, n_init=30)

    # Remove very correlated values
    # cor = np.corrcoef(data)
    # indexes = np.ones(data.shape[1])
    # for i in range(data.shape[1]):
    #     for j in range(i + 1, data.shape[1]):
    #         if cor[i, j] > 0.9:
    #             indexes[j] = 0
    # data = data[:, indexes.astype('bool')]

    layer_sizes = [7, 4, 3]
    dims = [6, 4, 3]
    evaluator = Evaluator(data, labels, 'silhouette', 'accuracy', 'ARI')

    if algorithm == 'gmm':
        alg = GMM(7, use_annealing=True, annealing_start_v=0.2,
                  plot_predictions=1, plot_wait_for_input=True,
                  init='random',
                  num_iter=100, update_rate=1,
                  evaluator=evaluator)
    elif algorithm == 'dgmm':
        alg = DGMM(layer_sizes, dims, init='kmeans', plot_predictions=10,
                   num_iter=100, num_samples=100,
                   update_rate=1e-4,
                   stopping_thresh=1e-4,
                   use_annealing=True, annealing_start_v=0.01,
                   evaluator=evaluator)
    elif algorithm == 'gmn':
        layer_sizes = [7, 12, 3]
        dims = [6, 4, 3]
        alg = GMN(layer_sizes, dims, init='kmeans',
                  plot_predictions=10,
                  update_rate=1e-4, stopping_thresh=0,
                  use_annealing=True, annealing_start_v=0.01,
                  num_iter=100, evaluator=evaluator)
    elif algorithm == 'gdgmm':
        alg = GDGMM([7, 6, 3], [6, 2, 1], init='kmeans', plot_predictions=10,
                    num_iter=100, step_size=0.01,
                    evaluator=evaluator)
    else:
        raise ValueError("Algorithm not supported")
    # alg = GMM(7, 7, init='kmeans', plot_predictions=False, num_iter=40)

    test_on_data(alg, data, labels, rescale=True)


def try_wine(algorithm='dgmm'):
    data, labels = datasets.load_wine(return_X_y=True)
    # 3 clusters, 13 dims
    evaluator = Evaluator(data, labels, 'silhouette', 'accuracy', 'ARI')

    if algorithm == 'kmeans':
        alg = KMeans(3)
    if algorithm == 'gmm':
        alg = GMM(3, init='kmeans', plot_predictions=False, num_iter=100,
                  evaluator=evaluator)
        # alg = GaussianMixture(3)
    elif algorithm == 'dgmm':
        alg = DGMM([3, 3, 3, 2], [10, 8, 5, 2], init='kmeans', plot_predictions=10,
                   num_iter=200, num_samples=1000,
                   update_rate=0.1,
                   stopping_thresh=1e-4,
                   use_annealing=True, annealing_start_v=0.01,
                   evaluator=evaluator)

    test_on_data(alg, data, labels)


def try_olive(algorithm='dgmm'):
    data = pd.read_csv('data/olive.csv')
    # 3 areas, 8 regions, 8 dims
    area, region = data.area, data.region
    labels = area.values
    data = data.iloc[:, 2:].values
    evaluator = Evaluator(data, labels, 'silhouette', 'accuracy', 'ARI')

    if algorithm == 'dgmm':
        alg = DGMM([3, 3, 3], [7, 6, 5], init='kmeans', plot_predictions=10,
                   num_iter=100,
                   num_samples=100,
                   use_annealing=True, annealing_start_v=0.1,
                   evaluator=evaluator)
    elif algorithm == 'gdgmm':
        alg = GDGMM([3, 2, 1], [7, 6, 5], init='kmeans', plot_predictions=10,
                    num_iter=1000, step_size=1e-5, num_samples=100,
                    evaluator=evaluator)
    else:
        raise ValueError("Algorithm not supported")
    # alg = GMM(7, 7, init='kmeans', plot_predictions=False, num_iter=40)

    test_on_data(alg, data, labels)


if __name__ == "__main__":
    algorithm = 'gmm'
    # try_manual(algorithm)
    try_ecoli(algorithm)
    # try_wine(algorithm)
    # try_olive(algorithm)
    pass
