import numpy as np
import math
from scipy.stats import multivariate_normal as normal
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


class GaussianDistrib:
    def __init__(self, dim, pi=None, eta=None, lambd=None, psi=None):
        if pi is None:
            pi = 1
        if eta is None:
            eta = np.ones(dim)
        if psi is None:
            psi = np.eye(dim)
        if lambd is None:
            lambd = np.eye(dim)

        # Parameters
        self.dim = dim
        self.pi = pi
        self.eta = eta
        self.lambd = lambd
        self.psi = psi

        # Values calculated during the EM
        self.pi_given_path = None
        self.mu_given_path = None
        self.sigma_given_path = None
        self.prob_theta_given_y = None

    def sample(self, value):
        if len(value.shape) == 1:
            u = normal.rvs(self.eta, self.psi, 1)[0]
            return u + self.lambd @ np.array([value]).T
        else:
            u = normal.rvs(self.eta, self.psi, len(value))
            return u + (self.lambd @ value.T).T


class DGMM:
    def __init__(self, layer_sizes, dims, plot_predictions=True,
                 init='kmeans'):
        def init_layer(layer_size, dim):
            pi = 1 / layer_size
            return [GaussianDistrib(dim, pi) for _ in range(layer_size)]

        self.dims = dims
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.layers = [init_layer(layer_sizes[i], dims[i])
                       for i in range(len(dims))]

        self.paths = self.get_paths_permutations(self.layer_sizes)

        # Display and computation parameters
        self.plot_predictions = plot_predictions
        if self.plot_predictions:
            plt.ion()
            fig, self.ax = plt.subplots(2, 1)
            self.ax[0].set_title("Predictions plot")
            self.ax[0].set_title("Distributions plot")
            plt.draw()
            plt.show(block=False)
        self.init = init

    def plot_prediction(self, data):
        COLORS = ['red', 'blue', 'green']

        probs = np.array([self.layers[0][dist_i].prob_theta_given_y
                         for dist_i in range(len(self.layers[0]))])
        clusters = np.argmax(probs, axis=0)
        sample, sample_clust = self.random_sample(100)

        self.ax[0].clear()
        self.ax[1].clear()
        for dist_i in range(len(self.layers[0])):
            values = data[clusters == dist_i]
            s_values = sample[sample_clust == dist_i]
            self.ax[0].plot(values[:, 0], values[:, 1], color=COLORS[dist_i],
                            label="cluster %d" % (dist_i + 1),
                            linestyle='', marker='.', markersize=8)
            self.ax[1].plot(s_values[:, 0], s_values[:, 1], color=COLORS[dist_i],
                            label="cluster %d" % (dist_i + 1),
                            linestyle='', marker='.', markersize=8)

        self.ax[0].legend()
        self.ax[1].legend()
        plt.draw()
        plt.pause(0.001)

    def compute_likelihood(self, data):
        # Start from last layer as standard normal
        pi = [1]
        sigma = [np.eye(self.dims[-1])]
        mu = [np.zeros(self.dims[-1])]

        # Move down to the first layer
        for layer_i in range(self.num_layers - 1, -1, -1):
            new_pi, new_mu, new_sigma = [], [], []

            for comb_current in range(self.layer_sizes[layer_i]):
                distrib = self.layers[layer_i][comb_current]
                distrib_pi, distrib_mu, distrib_sigma = [], [], []

                for comb_prev in range(len(sigma)):
                    # Compute new pi, mu and sigma
                    distrib_pi.append(distrib.pi * pi[comb_prev])
                    distrib_mu.append(distrib.eta + distrib.lambd @ mu[comb_prev])
                    sigma_i = distrib.psi + distrib.lambd @ sigma[comb_prev] @ distrib.lambd.T
                    sigma_i = self.make_spd(sigma_i)
                    distrib_sigma.append(sigma_i)

                # Save the values inside distribution
                distrib.pi_given_path = distrib_pi
                distrib.mu_given_path = distrib_mu
                distrib.sigma_given_path = distrib_sigma
                # Append to all the mus and sigmas
                new_pi += distrib_pi
                new_mu += distrib_mu
                new_sigma += distrib_sigma

            mu, sigma, pi = new_mu, new_sigma, new_pi

        prob_y_given_path = []
        prob_y_and_path = []
        for path_i in range(len(self.paths)):
            p_y_path = normal.pdf(data, mean=mu[path_i], cov=sigma[path_i])
            prob_y_given_path.append(p_y_path)
            prob_y_and_path.append(pi[path_i] * p_y_path)

        # WTF? why the original code rescaled 2 times
        # prob_y_and_path /= np.max(prob_y_and_path, axis=0, keepdims=True)

        # dims: [paths, num samples]
        prob_y_given_path = np.array(prob_y_given_path)
        prob_y_and_path = np.array(prob_y_and_path)

        prob_path_given_y = (prob_y_and_path / np.sum(prob_y_and_path, axis=0, keepdims=True))
        prob_y = np.sum(prob_y_and_path, axis=0)

        for layer_i in range(len(self.layer_sizes)):
            for dist_i in range(self.layer_sizes[layer_i]):
                # Sum over all the combinations where this distribution is
                # present to calculate the likelihood of its params
                index = self.paths[:, layer_i] == dist_i
                dist = self.layers[layer_i][dist_i]
                dist.prob_theta_given_y = prob_path_given_y[index].sum(axis=0)

        # s - most likelihood path for y
        # ps.y.list[layer, dist] <- prob_theta_given_y
        # hard.ps.y.list <- sampled

        if self.plot_predictions:
            self.plot_prediction(data)

        return prob_y, prob_y_given_path, prob_y_and_path, prob_path_given_y

    def _init_params(self, data):
        if self.init == 'random':
            # Initialization
            for dist_i in range(len(self.layers[0])):
                self.layers[0][dist_i].eta = data[np.random.choice(len(data))]
            for layer_i in range(1, self.num_layers):
                new_data = np.zeros([])
                for dist_i in range(len(self.layers[layer_i])):
                    dist = self.layers[layer_i][dist_i]
                    dist.eta = normal.rvs(
                        mean=np.zeros(self.dims[layer_i]),
                        cov=0.1 / self.num_layers ** 2,
                        size=1)[0]
                    dist.psi = np.eye(self.dims[layer_i]) / self.num_layers ** 2

            return

        if self.init == 'kmeans':
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=len(self.paths), max_iter=200, n_init=30)
            paths_i = kmeans.fit_predict(data)
        else:
            raise ValueError("Initialization '%s' is not supported" % self.init)

        from sklearn.decomposition import FactorAnalysis

        for layer_i in range(self.num_layers):
            for dist_i in range(len(self.layers[layer_i])):
                index = self.paths[paths_i][:, layer_i] == dist_i
                values = data[index]
                fa = FactorAnalysis(n_components=self.dims[layer_i],
                                    rotation='varimax')
                fa.fit(data)

                dist = self.layers[layer_i][dist_i]
                dist.eta = fa.mean_
                dist.lambd = fa.components_.T
                dist.psi = np.diag(fa.noise_variance_)

        # return np.arange(len(self.paths)) \
        #            .repeat(math.ceil(len(data) / len(self.paths)))[:len(data)]

    def fit(self, data, iter=10):
        inv = np.linalg.pinv
        num_observations = len(data)

        self._init_params(data)

        for iter_i in range(iter):
            prob_y, prob_y_given_path, prob_y_and_path, prob_path_given_y =\
                self.compute_likelihood(data)

            # Initialize the variables
            # Shape [num_observations, dim[layer_i - 1]]
            values_times_probs = data

            for layer_i in range(self.num_layers):
                layer = self.layers[layer_i]
                dim = self.dims[layer_i]

                # The combinations of lower layers are not included
                layer_paths_num = math.prod(self.layer_sizes[layer_i:])
                layer_paths = self.paths[:layer_paths_num]

                # Initialize variables that will be filled for each path
                rho_times_path_prob = np.zeros([len(layer), num_observations, dim])
                expect_zz_times_path_prob = np.zeros([len(layer), num_observations, dim, dim])
                z_times_path_prob = np.zeros([len(layer), num_observations, dim])

                for path_i in range(layer_paths_num):
                    path = layer_paths[path_i] + [0]

                    # Get the required parameters
                    dist_index = path[layer_i]
                    dist = layer[dist_index]
                    lambd, psi, eta = dist.lambd, dist.psi, dist.eta

                    if layer_i == self.num_layers - 1:
                        mu, sigma = np.zeros(self.dims[-1]), np.eye(self.dims[-1])
                    else:
                        dist_next = self.layers[layer_i + 1][path[layer_i + 1]]
                        path_i_next = path_i % math.prod(self.layer_sizes[layer_i + 2:])
                        mu = dist_next.mu_given_path[path_i_next]
                        sigma = dist_next.sigma_given_path[path_i_next]

                    # Estimate the parameters of the p(z[k+1] | z[k]) distribution
                    ksi = inv(inv(sigma) + lambd.T @ inv(psi) @ lambd)
                    ksi = self.make_spd(ksi)

                    # Shape [num_observations, dim[l-1]]
                    rho = (ksi @ (lambd.T @ inv(psi) @ (values_times_probs - eta).T
                                 + (inv(sigma) @ mu.reshape([-1, 1])))).T

                    # rho @ rho.T
                    rho_times_rho_T = np.array([r @ r.T for r in rho.reshape([*rho.shape, 1])])
                    # E(z z.T | s) = E^2(z | s) + Var(z | s)
                    expect_zz_given_path = rho_times_rho_T + ksi

                    # Sample from the distribution
                    z_sample = normal.rvs(cov=ksi, size=num_observations) + rho

                    # TODO rho_l, last layer

                    # Probability of the whole path given y
                    # Shape [num_observations, 1]
                    prob_path_given_y = dist.prob_theta_given_y * \
                        math.prod([self.layers[i][path[i]].prob_theta_given_y
                                   for i in range(layer_i + 1, len(path))])
                    prob_path_given_y = prob_path_given_y.reshape([-1, 1])

                    rho_times_path_prob[dist_index] += rho * prob_path_given_y
                    expect_zz_times_path_prob[dist_index] += expect_zz_given_path * prob_path_given_y.reshape([-1, 1, 1])
                    z_times_path_prob[dist_index] += z_sample * prob_path_given_y

                # Compute the best parameters estimate
                for dist_i in range(len(layer)):
                    dist = layer[dist_i]
                    rho_t_pp = rho_times_path_prob[dist_i]

                    probs = layer[dist_i].prob_theta_given_y.reshape([-1, 1])
                    denom = probs.sum()
                    # Shape [dim, dim]
                    expect_zz = np.sum(expect_zz_times_path_prob[dist_i], 0) / denom
                    normalized = (values_times_probs - dist.eta) * probs
                    lambd = normalized.T @ (rho_t_pp * probs) \
                        @ inv(expect_zz) / denom
                    psi = normalized.T @ normalized - \
                        normalized.T @ (rho_t_pp * probs) @ lambd / denom
                    eta = ((values_times_probs - (lambd @ rho_t_pp.T).T)
                           * probs / denom).sum(axis=0)
                    pi = probs.mean()

                    dist.pi, dist.eta, dist.lambd, dist.psi = pi, eta, lambd, psi

                # Fill the values for the next layer
                values_times_probs = z_times_path_prob.sum(0)

        self.compute_likelihood(data)
        return np.array([self.layers[0][dist_i].prob_theta_given_y
                         for dist_i in range(len(self.layers[0]))])

    def random_sample(self, num):
        values = []
        dists = []

        for i in range(num):
            value = normal.rvs(mean=np.zeros(self.dims[-1])).T
            dist = 0

            for layer_i in range(len(self.layers) - 1, -1, -1):
                layer = self.layers[layer_i]
                dist_i = np.random.choice(len(layer),
                    p=[layer[i].pi for i in range(len(layer))])
                dist = layer[dist_i]
                value = dist.lambd @ value + dist.eta + normal.rvs(cov=dist.psi)
                dist = dist_i

            values.append(value.reshape(-1))
            dists.append(dist)

        return np.array(values), np.array(dists)

    def predict(self, X):
        pass

    @staticmethod
    def make_spd(A):
        """
        Find the nearest-positive definite matrix to the given
        Source: https://itecnote.com/tecnote/python-convert-matrix-to-positive-semi-definite/
        """
        # symmetric
        A = 0.5 * (A + A.T)

        # positive definite
        _, s, V = np.linalg.svd(A)
        H = np.dot(V.T, np.dot(np.diag(s), V))
        A = (A + H) / 2
        A = (A + A.T) / 2

        if DGMM.is_pd(A):
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

        while not DGMM.is_pd(A):
            mineig = np.min(np.real(np.linalg.eigvals(A)))
            A += I * (-mineig * k ** 2 + spacing)
            k += 1

        return A

    @staticmethod
    def is_pd(A):
        """Returns true when input is positive-definite, via Cholesky"""
        try:
            _ = np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False

    @staticmethod
    def get_paths_permutations(layer_sizes):
        num = math.prod(layer_sizes)
        return np.array(
            [np.arange(num) // math.prod(layer_sizes[i + 1:]) % layer_sizes[i]
             for i in range(len(layer_sizes))]).T

