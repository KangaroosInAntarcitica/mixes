import pandas as pd
import numpy as np
from sklearn import metrics
from itertools import permutations


class Evaluator:
    @staticmethod
    def accuracy(labels, pred):
        m = metrics.confusion_matrix(labels, pred)
        i = [*permutations(np.arange(m.shape[1]))]
        all_perm = m[np.arange(m.shape[0]), i]
        return np.max(np.sum(all_perm, 1)) / len(labels)

    METRICS = {
        'accuracy': lambda data, labels, pred, probs: Evaluator.accuracy(labels, pred),
        'silhouette': lambda data, labels, pred, probs: metrics.silhouette_score(data, pred),
        'ARI': lambda data, labels, pred, probs: metrics.adjusted_rand_score(labels, pred),
        'RI': lambda data, labels, pred, probs: metrics.rand_score(labels, pred)
    }

    def __init__(self, data, labels=None, *metrics, show_log_lik=True,
                 print_metrics=True):
        self.show_log_lik = show_log_lik
        self.metrics = {}

        self.data = data
        self.labels = labels

        for metric in metrics:
            if metric in self.METRICS.keys():
                self.metrics[metric] = self.METRICS[metric]
            else:
                raise ValueError("Metric %s unsupported. Supported values: %s",
                                 (metric, sorted(self.METRICS.keys())))

        self.values = {k: [] for k in self.metrics.keys()}
        if self.show_log_lik:
            self.values['log_lik'] = []
        self.values['iter'] = []

        self.print_metrics = print_metrics

    def __call__(self, iter_i, probs, pred, log_lik):
        m = []

        self.values['iter'].append(iter_i)

        if self.show_log_lik:
            m.append("log_lik: %.5f" % log_lik)
            self.values['log_lik'].append(log_lik)

        for metric, func in self.metrics.items():
            value = func(self.data, self.labels, pred, probs)
            m.append("%s: %.5f" % (metric, value))
            self.values[metric].append(value)

        if self.print_metrics:
            text = ", ".join(m)
            text = "Iter %3d (%s)" % (iter_i, text)
            print(text)

    def get(self, key):
        if key not in self.values:
            raise KeyError("There is no metric '%s' collected" % key)
        if len(self.values[key]) < 1:
            raise RuntimeError("Evaluator has not collected any metrics")

        return self.values[key][-1]

    def get_values(self):
        return self.values

    def get_dataframe(self):
        return pd.DataFrame(self.values).set_index('iter')
