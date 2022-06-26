import numpy as np
from scipy.stats import multivariate_normal as normal
import matplotlib.pyplot as plt
from dgmm import SamplingDGMM as DGMM
from dgmm import GMM
from sklearn import metrics, preprocessing, datasets
from sklearn.cluster import KMeans
import pandas as pd
from itertools import permutations


def accuracy(labels, pred):
    m = metrics.confusion_matrix(labels, pred)
    i = [*permutations(np.arange(m.shape[1]))]
    all_perm = m[np.arange(m.shape[0]), i]
    return np.max(np.sum(all_perm, 1)) / len(labels)


def create_evaluator(data, labels):
    silhouette_real = metrics.silhouette_score(data, labels)

    def evaluator(iter_i, probs, clusters, log_lik):
        silhouette = metrics.silhouette_score(data, clusters)
        acc = accuracy(labels, clusters)
        ari = metrics.adjusted_rand_score(labels, clusters)

        print("Iter %3d: (sil: %.3f / %.3f, acc: %.3f, ARI: %.3f, log_lik: %.5f)" %\
              (iter_i + 1, silhouette, silhouette_real, acc, ari, log_lik))

    return evaluator


def test_on_data(alg, data, labels, rescale=True):
    if rescale:
        data = preprocessing.scale(data)

    if hasattr(alg, "fit_predict"):
        alg.fit_predict(data)
        clust = alg.labels_
    else:
        clust = alg.fit(data)

    print()
    print("Silhouette score = ", metrics.silhouette_score(data, labels))
    print("Accuracy = ", accuracy(labels, clust))
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


def try_manual(algorithm='dgmm'):
    data, labels = manual_dataset_2()
    if algorithm == 'dgmm':
        alg = DGMM([2], [3], init='kmeans', plot_predictions=1,
                   plot_wait_for_input=True,
                   num_iter=100, num_samples=5000,
                   evaluator=create_evaluator(data, labels))
    elif algorithm == 'gmm':
        alg = GMM(2, init='random', plot_predictions=1, num_iter=40,
                  plot_wait_for_input=True,
                  evaluator=create_evaluator(data, labels))
    elif algorithm == 'kmeans':
        alg = KMeans(3, max_iter=100, n_init=30)

    test_on_data(alg, data, labels, rescale=False)


def try_ecoli():
    data, labels = load_ecoli()
    # alg = KMeans(3, max_iter=100, n_init=30)
    alg = DGMM([7, 4, 3, 2], [4, 2, 2, 2], init='kmeans', plot_predictions=20,
               num_iter=100, num_samples=1000,
               evaluator=create_evaluator(data, labels))
    # alg = GMM(7, 7, init='kmeans', plot_predictions=False, num_iter=40)

    test_on_data(alg, data, labels)


def try_wine():
    data, labels = datasets.load_wine(return_X_y=True)
    alg = DGMM([3, 3, 3], [6, 4, 2], init='kmeans', plot_predictions=20,
               num_iter=100)
    # alg = GMM(7, 7, init='kmeans', plot_predictions=False, num_iter=40)

    test_on_data(alg, data, labels)


if __name__ == "__main__":
    try_manual()
    # try_ecoli()
    # try_wine()
