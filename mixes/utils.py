import numpy as np
import math
import warnings


SMALL_VALUE = 1e-20


def get_paths_permutations(layer_sizes):
    num = math.prod(layer_sizes)
    return np.array(
        [np.arange(num) // math.prod(layer_sizes[i + 1:]) % layer_sizes[i]
         for i in range(len(layer_sizes))]).T


class GaussianDistrib:
    def __init__(self, tau=None, eta=None, lambd=None, psi=None):
        if tau is None:
            tau = 1

        # Parameters
        self.tau = tau
        self.eta = eta
        self.lambd = lambd
        self.psi = psi

        # Values calculated during the EM
        self.pi_given_path = None
        self.mu_given_path = None
        self.sigma_given_path = None
        self.prob_theta_given_y = None


def make_spd(A):
    """
    Find the nearest symmetric positive definite (SPD) matrix to the given
    Source: https://itecnote.com/tecnote/python-convert-matrix-to-positive-semi-definite/
    """
    # symmetric
    A = 0.5 * (A + A.T)

    # positive definite
    _, s, V = np.linalg.svd(A)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A = (A + H) / 2
    A = (A + A.T) / 2

    if is_pd(A):
        return A

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1

    while not is_pd(A):
        mineig = np.min(np.real(np.linalg.eigvals(A)))
        A += I * (-mineig * k ** 2 + spacing)
        k += 1

    return A


def is_pd(A):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def was_stopping_criterion_reached(log_likelihoods, stopping_thresh):
    """
    :param log_likelihoods: list of log likelihoods
    :param stopping_thresh: the stopping threshold
    :return: whether the stopping criterion was reached
    """
    log_lik = log_likelihoods
    if len(log_lik) >= 3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aitken_acceleration = (log_lik[-1] - log_lik[-2]) / \
                                  (log_lik[-2] - log_lik[-3])
            l_inf = log_lik[-2] + (log_lik[-1] - log_lik[-2]) / \
                    (1 - aitken_acceleration)
            if np.abs(l_inf - log_lik[-1]) < stopping_thresh:
                return True
    return False
