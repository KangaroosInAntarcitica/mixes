import numpy as np
import scipy as sp
from scipy import stats
from scipy.stats import multivariate_normal as normal
import matplotlib.pyplot as plt
import seaborn as sns
import math
from DGMM import DGMM
from sklearn import metrics, datasets, preprocessing
from sklearn.cluster import KMeans
import pandas as pd


def test_on_data(alg, data, labels):
    scaler = preprocessing.StandardScaler()
    data = scaler.fit_transform(data)

    if hasattr(alg, "fit_predict"):
        alg.fit_predict(data)
        clust = alg.labels_
    else:
        probs = alg.fit(data)
        clust = np.argmax(probs, axis=1)

    print("Silhouette score = ", metrics.silhouette_score(data, labels))
    print("Accuracy = ", metrics.accuracy_score(labels, clust))
    print("ARI = ", metrics.adjusted_rand_score(labels, clust))


def load_ecoli():
    data = pd.read_csv('data/ecoli.csv', header=None)
    data, labels = data.values[:,:-1].astype('float'), data.values[:,-1]
    mapping = {v: i for i, v in enumerate(np.unique(labels))}
    labels = np.array([mapping[x] for x in labels])
    return data, labels


def manual_dataset():
    l11 = normal([0, 0, 0])
    l12 = normal([5, 5, 0])
    l13 = normal([-5, 7, 0])
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
        np.ones(500) * 3
    ])

    return data, labels


# data, labels = datasets.load_wine(return_X_y=True)
data, labels = load_ecoli()
alg = KMeans(3, max_iter=100, n_init=30)
alg = DGMM([7, 5, 3], [7, 6, 2], init='kmeans', plot_predictions=True, num_iter=30)

# data, labels = manual_dataset()
# alg = KMeans(3, max_iter=100, n_init=30)
# alg = DGMM([3, 5], [3, 2], init='kmeans', plot_predictions=True, num_iter=10)

test_on_data(alg, data, labels)

try:
    plt.pause(1e10)
except Exception as e:
    pass

if __name__ == "__main__":
    pass

