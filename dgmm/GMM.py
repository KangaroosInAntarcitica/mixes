from .AbstractDGMM import *


class GMM(AbstractDGMM):
    def __init__(self, num_dists, plot_predictions=True,
                 plot_wait_for_input=True, init='kmeans',
                 num_iter=10, evaluator=None):
        super().__init__([num_dists], [1],
                         plot_predictions=plot_predictions,
                         plot_wait_for_input=plot_wait_for_input,
                         init=init,
                         num_iter=num_iter,
                         evaluator=evaluator)

    def compute_likelihood(self, data, iter_i):
        """
        :return: probability of data
        """
        n_dists = self.layer_sizes[0]
        dists = self.layers[0]

        probs_sum = 0
        for dist_i in range(n_dists):
            dist = dists[dist_i]
            # Set the variables required for optimization
            dist.prob_theta_given_y = \
                dist.pi * normal.pdf(data, mean=dist.eta, cov=dist.psi)
            probs_sum += dist.prob_theta_given_y
            # Set variables required for plotting
            dist.mu_given_path = [dist.eta]
            dist.sigma_given_path = [dist.psi]

        for dist_i in range(n_dists):
            # Normalize
            dists[dist_i].prob_theta_given_y /= probs_sum

        if iter_i is not None:
            self.evaluate(data, iter_i)

        return probs_sum

    def predict_path_probs(self, data):
        probs_v = self.compute_likelihood(data, None)
        dist_probs = np.array([self.layers[0][dist_i].prob_theta_given_y
                               for dist_i in range(len(self.layers[0]))]).T
        return dist_probs.T, dist_probs.T, probs_v

    def fit(self, data):
        assert self.num_layers == 1, "GMM only allows one layer"

        self.out_dims = [data.shape[1]]
        self._init_params(data)

        n_dists = self.layer_sizes[0]
        dists = self.layers[0]

        for iter_i in range(self.num_iter):
            # Expectation step
            self.compute_likelihood(data, iter_i)

            # Maximization step
            for dist_i in range(n_dists):
                dist = dists[dist_i]
                probs = dist.prob_theta_given_y.reshape([-1, 1])

                denom = np.sum(probs)
                eta = (data * probs).sum(0) / denom
                psi = np.sum([(data[i:i+1] - eta).T @ (data[i:i+1] - eta) *
                              probs[i] for i in range(len(data))], 0) / denom
                pi = denom / len(data)

                dist.eta, dist.psi, dist.pi = eta, psi, pi
                dist.lambd = np.zeros([self.out_dims[0], 1])

        self.compute_likelihood(data, self.num_iter)
        return self.predict(data)

    def random_sample(self, num):
        values = []
        clusters = []
        dists = self.layers[0]

        for i in range(num):
            dist_i = np.random.choice(
                len(dists), p=[dists[i].pi for i in range(len(dists))])
            value = normal.rvs(mean=dists[dist_i].eta, cov=dists[dist_i].psi)\
                    * np.array([1])

            values.append(value.reshape(-1))
            clusters.append(dist_i)

        return np.array(values), np.array(clusters)
