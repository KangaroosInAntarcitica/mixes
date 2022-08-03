import warnings
import numpy as np
from sklearn import metrics


def create_log_lik_criterion(stopping_thresh=1e-2):
    log_lik_values = []

    def criterion(iter_i, data, probs, pred, log_lik):
        # Add average log likelihood
        log_lik_values.append(log_lik / len(data))
        return __aitken_acceleration_criterion(log_lik_values, stopping_thresh)

    return criterion


def create_silhouette_criterion(stopping_thresh=1e-2):
    silhouette_scores = []

    def criterion(iter_i, data, probs, pred, log_lik):
        silhouette_scores.append(metrics.silhouette_score(data, pred))
        return __aitken_acceleration_criterion(
            silhouette_scores, stopping_thresh)

    return criterion


def __aitken_acceleration_criterion(values, stopping_thresh):
    """
    :param values: list of values
    :param stopping_thresh: the stopping threshold
    :return: whether the stopping criterion was reached
    """
    if len(values) >= 3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aitken_acceleration = (values[-1] - values[-2]) / \
                                  (values[-2] - values[-3])
            l_inf = values[-2] + (values[-1] - values[-2]) / \
                    (1 - aitken_acceleration)
            if np.abs(l_inf - values[-1]) < stopping_thresh:
                return True
    return False