from .AbstractDGMM import *


class GMM(AbstractDGMM):
    def __init__(self, num_dists, *args, **kwargs):
        super().__init__([num_dists], [1], *args, **kwargs)

    def compute_likelihood(self, data, annealing_v=1):
        """
        :return: probability of data
        """
        n_dists = self.layer_sizes[0]
        dists = self.layers[0]

        probs_sum = 0
        for dist_i in range(n_dists):
            dist = dists[dist_i]
            # Set the variables required for optimization
            sigma = dist.sigma_given_path[0]

            dist.prob_theta_given_y = \
                dist.pi * normal.pdf(data, mean=dist.eta,
                                     cov=sigma, allow_singular=True)
            if annealing_v != 1:
                dist.prob_theta_given_y **= annealing_v
            probs_sum += dist.prob_theta_given_y

        for dist_i in range(n_dists):
            # Normalize
            dists[dist_i].prob_theta_given_y /= (probs_sum + self.SMALL_VALUE)

        return probs_sum

    def predict_path_probs(self, data, annealing_value=1):
        probs_v = self.compute_likelihood(data, annealing_value)
        dist_probs = np.array([self.layers[0][dist_i].prob_theta_given_y
                               for dist_i in range(len(self.layers[0]))]).T
        return dist_probs.T, dist_probs.T, probs_v

    def fit(self, data):
        assert self.num_layers == 1, "GMM only allows one layer"

        n_dists = self.layer_sizes[0]
        dists = self.layers[0]

        self.out_dims = [data.shape[1]]
        self.in_dims = [data.shape[1]]
        # Initialize the paramter and set them correctly for GMM
        self._init_params(data)

        for dist_i in range(n_dists):
            dist = dists[dist_i]
            dist.sigma_given_path = [dist.psi + dist.lambd @ dist.lambd.T]
            dist.mu_given_path = [dist.eta]

        for iter_i in range(self.num_iter):
            # Expectation step
            self.compute_likelihood(data, self.annealing_v)
            self.evaluate(data, iter_i)

            pis_sum = 0

            # Maximization step
            for dist_i in range(n_dists):
                dist = dists[dist_i]

                probs = dist.prob_theta_given_y.reshape([-1, 1])

                denom = np.sum(probs)
                eta = (data * probs).sum(0) / denom
                sigma = ((data - eta) * probs).T @ (data - eta) / denom
                pi = denom / len(data)

                # Update the parameters
                rate = self.update_rate
                eta = eta * rate + dist.eta * (1 - rate)
                sigma = sigma * rate + dist.sigma_given_path[0] * (1 - rate)
                pi = pi * rate + dist.pi * (1 - rate)
                pis_sum += pi

                # Perform certain steps to make it consistent with the
                #   abstract dgmm. As a result sigma = psi + lambda @ lambd.T
                psi = np.eye(data.shape[1]) * 0
                u, s, vh = np.linalg.svd(sigma, hermitian=True)
                lambd = u @ np.diag(np.sqrt(s))

                # Set the values
                dist.eta, dist.lambd, dist.psi, dist.pi = eta, lambd, psi, pi
                dist.sigma_given_path = [psi + lambd @ lambd.T]
                dist.mu_given_path = [eta]

            for dist_i in range(n_dists):
                dists[dist_i].pi /= pis_sum

            if self.use_annealing:
                self.annealing_v = np.clip(self.annealing_v + self.annealing_step, 0, 1)

        self.compute_likelihood(data)
        self.evaluate(data, self.num_iter)
        return self.predict(data)

    def random_sample(self, num):
        values = []
        clusters = []
        dists = self.layers[0]

        for i in range(num):
            dist_i = np.random.choice(
                len(dists), p=[dists[i].pi for i in range(len(dists))])
            dist = dists[dist_i]
            cov = dist.sigma_given_path[0]
            value = normal.rvs(mean=dist.eta, cov=cov) * np.array([1])

            values.append(value.reshape(-1))
            clusters.append(dist_i)

        return np.array(values), np.array(clusters)
