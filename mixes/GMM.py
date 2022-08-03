from scipy.stats import multivariate_normal as normal
from .utils import *


class GMM:
    def __init__(self, num_dists,
                 init='kmeans', num_iter=10,
                 use_annealing=False, annealing_start_v=0.1, update_rate=0.1,
                 evaluator=None, stopping_criterion=None,
                 var_regularization=1e-6):
        self.num_dists = num_dists
        self.dists = [self.GaussDist(tau=1 / self.num_dists)
                      for i in range(self.num_dists)]

        self.init = init
        self.num_iter = num_iter

        self.use_annealing = use_annealing
        self.annealing_v = annealing_start_v if use_annealing else 1
        self.annealing_step = (1 - self.annealing_v) / \
                              (self.num_iter + SMALL_VALUE) / 0.9

        self.update_rate = update_rate
        self.evaluator = evaluator
        self.stopping_criterion = stopping_criterion
        self.var_regularization = var_regularization

    def predict_path_probs(self, data, annealing_v: float = 1):
        """
        :return: probability of data
        """
        log_prob_v_and_dist = []
        for dist_i in range(self.num_dists):
            dist = self.dists[dist_i]

            log_prob = np.log(dist.tau)
            log_prob += normal.logpdf(data, mean=dist.mu, cov=dist.sigma,
                                      allow_singular=True)
            if annealing_v != 1:
                log_prob *= annealing_v
            log_prob_v_and_dist.append(log_prob)

        # dim [dist, v]
        log_prob_v_and_dist = np.array(log_prob_v_and_dist)

        # Rescale the variables for numerical stability
        log_prob_v_and_dist_max = np.max(log_prob_v_and_dist, axis=0)
        log_prob_v_and_dist -= log_prob_v_and_dist_max
        prob_v_and_dist = np.exp(log_prob_v_and_dist)
        prob_v = np.sum(prob_v_and_dist, axis=0)
        # Use the Bayes formula p(path|v) = p(v,path) / p(v)
        prob_dist_given_v = prob_v_and_dist / (prob_v + SMALL_VALUE)
        prob_v *= np.exp(log_prob_v_and_dist_max)

        return prob_dist_given_v, prob_v

    def predict(self, data, probs=False):
        prob_dist_given_v, _ = self.predict_path_probs(data)
        return prob_dist_given_v.T if probs else \
            np.argmax(prob_dist_given_v, 0)

    def evaluate(self, data, iter_i: int):
        probs, prob_v = self.predict_path_probs(data)
        clusters = np.argmax(probs, 0)
        log_lik = np.sum(np.log(prob_v)) if np.all(prob_v != 0) else -np.inf

        if self.evaluator is not None:
            self.evaluator(iter_i, data, probs.T, clusters, log_lik)

        stopping_criterion_reached = \
            self.stopping_criterion(iter_i, data, probs.T, clusters, log_lik) \
            if self.stopping_criterion is not None else False

        return stopping_criterion_reached

    def _init_params(self, data):
        if self.init == 'kmeans':
            from sklearn.cluster import KMeans
            kmeans = KMeans(self.num_dists)
            clusters = kmeans.fit_predict(data)
            clusters_i = np.unique(clusters)

            for i in range(self.num_dists):
                values = data[clusters == clusters_i[i]]
                dist = self.dists[i]
                dist.mu = kmeans.cluster_centers_[i]
                sigma = np.cov(values.T)
                dist.sigma = sigma
                dist.tau = np.sum(clusters == clusters_i[i]) / len(clusters)
        else:
            raise ValueError("Initialization '%s' is not supported" % self.init)

    def fit(self, data):
        # Initialize the parameters
        self._init_params(data)
        self.evaluate(data, 0)

        for iter_i in range(self.num_iter):
            # Expectation step
            prob_dist_given_v, prob_v = \
                self.predict_path_probs(data, self.annealing_v)

            tau_sum = 0

            # Maximization step
            for dist_i in range(self.num_dists):
                dist = self.dists[dist_i]
                probs = prob_dist_given_v[dist_i].reshape([-1, 1])

                denom = np.sum(probs)
                mu = (data * probs).sum(0) / denom
                sigma = ((data - mu) * probs).T @ (data - mu) / denom
                tau = denom / len(data)

                sigma += np.eye(len(sigma)) * self.var_regularization

                # Update the parameters
                rate = self.update_rate
                mu = mu * rate + dist.mu * (1 - rate)
                sigma = sigma * rate + dist.sigma * (1 - rate)
                tau = tau * rate + dist.tau * (1 - rate)
                tau_sum += tau

                # Set the values
                dist.tau, dist.mu, dist.sigma = tau, mu, sigma

            for dist_i in range(self.num_dists):
                self.dists[dist_i].tau /= tau_sum

            stopping_criterion_reached = self.evaluate(data, iter_i + 1)
            if stopping_criterion_reached:
                break

            if self.use_annealing:
                self.annealing_v = self.annealing_v + self.annealing_step
                self.annealing_v = float(np.clip(self.annealing_v, 0, 1))

        return self.predict(data)

    def random_sample(self, num):
        values = []
        clusters = []

        for i in range(num):
            dist_i = np.random.choice(
                self.num_dists,
                p=[self.dists[i].tau for i in range(self.num_dists)])
            dist = self.dists[dist_i]
            value = normal.rvs(mean=dist.mu, cov=dist.sigma) * np.array([1])

            values.append(value.reshape(-1))
            clusters.append(dist_i)

        return np.array(values), np.array(clusters)

    class GaussDist:
        def __init__(self, tau=None, mu=None, sigma=None):
            self.tau = tau
            self.mu = mu
            self.sigma = sigma

    @staticmethod
    def _calculate_lambda_psi(sigma):
        """
        Calculate lambda and psi if needed to convert to a dgmm format
            As a result sigma = psi + lambda @ lambd.T
        :param sigma: sigma
        :return: lambda, psi
        """
        psi = np.eye(sigma.shape[0]) * 0
        u, s, vh = np.linalg.svd(sigma, hermitian=True)
        lambd = u @ np.diag(np.sqrt(s))
        return lambd, psi
