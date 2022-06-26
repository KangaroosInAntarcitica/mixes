import numpy as np
import math
from scipy.stats import multivariate_normal as normal
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches
import matplotlib
matplotlib.use("TkAgg")

from .GaussianDistrib import GaussianDistrib


class AbstractDGMM:
    SMALL_VALUE = 1e-20

    def __init__(self, layer_sizes, dims, plot_predictions=True,
                 plot_wait_for_input=False,
                 init='kmeans', num_iter=10, num_samples=500,
                 evaluator=None):
        def init_layer(layer_size, dim):
            pi = 1 / layer_size
            return [GaussianDistrib(dim, pi) for _ in range(layer_size)]

        # Dimensions on the input (next layer)
        self.in_dims = dims
        # Dimensions on the output (prev layer)
        # First value will be the dimensions of the data
        self.out_dims = [0] + dims[:-1]

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.layers = [init_layer(layer_sizes[i], dims[i])
                       for i in range(len(dims))]
        self.num_iter = num_iter
        self.num_samples = num_samples

        self.paths = self.get_paths_permutations(self.layer_sizes)

        # Display and computation parameters
        self.plot_wait_for_intput = plot_wait_for_input
        self.plot_predictions = int(plot_predictions)
        if self.plot_predictions:
            plt.ion()
            self.fig, self.ax = plt.subplots(2, 1)
            self.ax[0].set_title("Predictions plot")
            self.ax[0].set_title("Distributions plot")
            plt.draw()
            plt.show(block=False)

        self.init = init
        self.evaluator = evaluator

    def fit(self, data):
        raise NotImplementedError("Fit is not implemented in the abstract class")

    def compute_path_distributions(self):
        """
        Compute the distributions of forward parameters for all the paths
        This represents the mu, sigma, and pi parameters

        Also updates the layers[layer_i][dist_i].mu_given_path,
        .sigma_given_path and .pi_given_path properties of distributions to
        correspond to paths starting from this distribution

        :return: (mu, sigma, pi) - The list of parameters for all the paths
        """
        # The variables pi, sigma, mu will hold the values for each path
        #   starting from the last layer to a given layer. There is only one
        #   path at the last layer - standard normal distribution
        pi = [1]
        sigma = [np.eye(self.in_dims[-1])]
        mu = [np.zeros(self.in_dims[-1])]

        # Move down from the last to the first layer
        for layer_i in range(self.num_layers - 1, -1, -1):
            new_pi, new_mu, new_sigma = [], [], []

            # Loop through distributions of this layer
            for dist_i in range(self.layer_sizes[layer_i]):
                dist = self.layers[layer_i][dist_i]
                dist_pi, dist_mu, dist_sigma = [], [], []

                # Loop through the paths at the previous layer and extend them
                #   with the current distribution
                # Note: this is the same order that self.paths are generated
                #   (upper layers repeated for each dist_i of lower layers)
                for prev_path in range(len(sigma)):
                    # Compute new pi, mu and sigma (also make it spd)
                    dist_pi.append(dist.pi * pi[prev_path])
                    dist_mu.append(dist.eta + dist.lambd @ mu[prev_path])
                    sigma_i = dist.psi + dist.lambd @ sigma[
                        prev_path] @ dist.lambd.T
                    dist_sigma.append(self.make_spd(sigma_i))

                # Save the values inside distribution
                dist.pi_given_path = dist_pi
                dist.mu_given_path = dist_mu
                dist.sigma_given_path = dist_sigma
                # Append to all the mus and sigmas (at current layer)
                new_pi += dist_pi
                new_mu += dist_mu
                new_sigma += dist_sigma

            mu, sigma, pi = new_mu, new_sigma, new_pi

        return mu, sigma, pi

    def compute_paths_prob_given_out_values(self, values, layer_i: int):
        """
        At a specific layer compute the probability of output values given the
        current distribution parameters
        Note that compute_path_distributions needs to be called to update
        the values of mu, sigma and pi for each distribution

        :param values: The out values
        :param layer_i: The layer number staring from 0 to self.num_layers - 1
        :return: probability for each path given value, array of shape
            [len(paths), len(values)]
        """
        layer = self.layers[layer_i]

        # Initialize the arrays to store values in
        prob_v_and_path = []
        for dist_i in range(len(layer)):
            # Get the values from the distribution
            mu = layer[dist_i].mu_given_path
            sigma = layer[dist_i].sigma_given_path
            pi = layer[dist_i].pi_given_path

            for path_i in range(len(mu)):
                p = normal.pdf(values, mean=mu[path_i],
                               cov=sigma[path_i], allow_singular=True)
                prob_v_and_path.append(p * pi[path_i])

        prob_v_and_path = np.array(prob_v_and_path)
        prob_v = np.sum(prob_v_and_path, 0)
        # Use the Bayes formula p(path|v) = p(v,path) / p(v)
        prob_path_given_v = prob_v_and_path / (prob_v + self.SMALL_VALUE)

        return prob_v, prob_path_given_v

    def predict_path_probs(self, data):
        prob_v, prob_path_given_v = \
            self.compute_paths_prob_given_out_values(data, 0)
        n_clusters = self.layer_sizes[0]
        prob_dist_given_v = prob_path_given_v \
            .reshape([n_clusters, -1, len(data)]).sum(axis=1)
        return prob_path_given_v, prob_dist_given_v, prob_v

    def predict(self, data, probs=False):
        _, prob_dist_given_v, _ = self.predict_path_probs(data)

        if probs:
            return prob_dist_given_v.T
        else:
            return np.argmax(prob_dist_given_v, 0)

    def _init_params(self, data):
        if self.init == 'random':
            # Initialization
            for layer_i in range(self.num_layers):
                for dist_i in range(self.layer_sizes[layer_i]):
                    dist = self.layers[layer_i][dist_i]
                    dim, in_dim = self.out_dims[layer_i], self.in_dims[layer_i]

                    if layer_i == 0:
                        dist.eta = data[np.random.choice(len(data))]
                    else:
                        dist.eta = normal.rvs(mean=np.zeros(dim), cov=0.1)
                    dist.psi = np.eye(dim) / self.num_layers ** 2
                    dist.lambd = np.random.random([dim, in_dim]) / self.num_layers ** 2
                    dist.lambd /= dist.lambd.sum(axis=0, keepdims=True)
                    dist.pi = 1 / self.layer_sizes[layer_i]
            return

        if self.init == 'kmeans-1':
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.layer_sizes[0])
            kmeans.fit_predict(data)

            for layer_i in range(self.num_layers):
                for dist_i in range(self.layer_sizes[layer_i]):
                    dist = self.layers[layer_i][dist_i]
                    dim, in_dim = self.out_dims[layer_i], self.in_dims[layer_i]

                    if layer_i == 0:
                        dist.eta = kmeans.cluster_centers_[dist_i]
                    else:
                        dist.eta = normal.rvs(mean=np.zeros(dim), cov=3)
                    dist.psi = np.eye(dim) / 4 / self.num_layers ** 2
                    dist.lambd = -1 + 2 * np.random.random([dim, in_dim])
                    dist.lambd /= np.apply_along_axis(
                        np.linalg.norm, 1, dist.lambd).reshape([-1, 1])
                    dist.pi = 1 / self.layer_sizes[layer_i]
            return

        if self.init == 'deep-kmeans':
            from sklearn.cluster import KMeans

            values = np.copy(data)
            for layer_i in range(self.num_layers):
                kmeans = KMeans(n_clusters=self.layer_sizes[layer_i])
                kmeans.fit_predict(values)
                labels = np.unique(kmeans.labels_)

                for dist_i in range(self.layer_sizes[layer_i]):
                    dist_values = values[kmeans.labels_ == labels[dist_i]]
                    dist = self.layers[layer_i][dist_i]
                    dim, in_dim = self.out_dims[layer_i], self.in_dims[layer_i]

                    dist.eta = kmeans.cluster_centers_[dist_i]
                    dist.psi = np.diag(np.var(dist_values, axis=0))
                    dist.lambd = -1 + 2 * np.random.random([dim, in_dim])
                    dist.lambd /= np.apply_along_axis(
                        np.linalg.norm, 1, dist.lambd).reshape([-1, 1])
                    dist.pi = len(dist_values) / len(values)

                    values[kmeans.labels_ == labels[dist_i]] = \
                        (np.linalg.pinv(dist.lambd) @
                        (np.linalg.pinv(dist.psi) @ dist_values.T -
                         dist.eta.reshape([-1, 1]))).T
            return

        if self.init == 'kmeans':
            from sklearn.cluster import KMeans
            from sklearn.decomposition import FactorAnalysis
            import warnings

            values = data

            for layer_i in range(self.num_layers):
                kmeans = KMeans(n_clusters=self.layer_sizes[layer_i],
                                n_init=30, max_iter=200)
                clusters = kmeans.fit_predict(values)
                cluster_i = np.unique(clusters)
                next_values = np.zeros([len(values), self.in_dims[layer_i]])

                # c1 = clusters == cluster_i[0]
                # c2 = clusters == cluster_i[1]
                # self.ax[1].clear()
                # plt.plot(values[c1, 0], values[c1, 1], 'r.')
                # plt.plot(values[c2, 0], values[c2, 1], 'b.')
                # self.ax[1].set_aspect('equal', 'box')
                # plt.show()

                for dist_i in range(self.layer_sizes[layer_i]):
                    index = clusters == cluster_i[dist_i]
                    values_i = values[index]
                    fa = FactorAnalysis(n_components=self.in_dims[layer_i],
                                        rotation='varimax')
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        next_values_i = fa.fit_transform(values_i)
                    # next_values[index] = (np.linalg.pinv(fa.components_.T) @
                    #                       (values_i - fa.mean_).T).T
                    next_values[index] = next_values_i

                    dist = self.layers[layer_i][dist_i]
                    dist.eta = fa.mean_

                    dist.lambd = fa.components_.T
                    if fa.components_.shape[0] != fa.n_components:
                        dist.lambd = np.repeat(dist.lambd, fa.n_components, axis=1)
                    dist.psi = np.diag(fa.noise_variance_)
                    dist.pi = 1 / self.layer_sizes[layer_i]

                values = next_values
        else:
            raise ValueError("Initialization '%s' is not supported" % self.init)

        # from sklearn.decomposition import FactorAnalysis
        #
        # for layer_i in range(self.num_layers):
        #     next_data = np.zeros([len(data), self.dims[layer_i]])
        #
        #     for dist_i in range(self.layer_sizes[layer_i]):
        #         index = self.paths[paths_i][:, layer_i] == dist_i
        #         values = data[index]
        #         fa = FactorAnalysis(n_components=self.dims[layer_i],
        #                             rotation='varimax')
        #         fa.fit(values)
        #
        #         dist = self.layers[layer_i][dist_i]
        #         dist.eta = fa.mean_
        #         dist.lambd = fa.components_.T
        #         dist.psi = np.diag(fa.noise_variance_)
        #         dist.pi = 1 / self.layer_sizes[layer_i]
        #
        #         next_data[index] = fa.transform(values)
        #
        #     data = next_data

    def evaluate(self, data, iter_i):
        _, probs, prob_v = self.predict_path_probs(data)
        clusters = np.argmax(probs, 0)

        if self.evaluator is not None:
            log_lik = np.sum(np.log(prob_v))
            self.evaluator(iter_i, probs.T, clusters, log_lik)

        if not self.plot_predictions or (iter_i % self.plot_predictions != 0):
            return

        def draw_distribution(mean, cov, ax, color):
            # How many sigmas to draw. 2 sigmas is >95%
            N_SIGMA = 2
            mean, cov = mean[:2], cov[:2,:2]
            # Since covariance is SPD, svd will produce orthogonal eigenvectors
            U, S, V = np.linalg.svd(cov)
            # Calculate the angle of first eigenvector
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))

            # Eigenvalues are now half-width and half-height squared
            std = np.sqrt(S)
            ax.add_patch(matplotlib.patches.Ellipse(mean, 2 * std[0] * N_SIGMA,
                2 * std[1] * N_SIGMA, angle, color=[*color[:3], 0.3], linewidth=0))

        sample, sample_clust = self.random_sample(200)

        colors = cm.rainbow(np.linspace(0, 1, probs.shape[0]))

        # Draw the predictions plot
        self.ax[0].clear()
        data_colors = np.clip(probs.T @ colors, 0, 1)
        self.ax[0].scatter(data[:, 0], data[:, 1], color=data_colors, s=10)
        self.ax[0].set_aspect('equal', 'box')
        self.ax[0].set_title("Probabilities")

        # Draw the sample plot
        self.ax[1].clear()
        for dist_i in range(self.layer_sizes[0]):
            s_values = sample[sample_clust == dist_i]
            color = colors[dist_i]
            self.ax[1].scatter(s_values[:, 0], s_values[:, 1], color=color,
                               label="cluster %d" % (dist_i + 1), s=10)
            dist = self.layers[0][dist_i]
            for path_i in range(len(dist.mu_given_path)):
                draw_distribution(dist.mu_given_path[path_i],
                                  dist.sigma_given_path[path_i],
                                  self.ax[1], color)
        self.ax[1].set_xlim(self.ax[0].get_xlim())
        self.ax[1].set_ylim(self.ax[0].get_ylim())
        self.ax[1].set_aspect('equal', 'box')
        self.ax[1].set_title("Sample")
        self.ax[1].legend()

        # Draw the plots
        self.fig.suptitle("Iteration %d" % iter_i)
        plt.draw()

        if self.plot_wait_for_intput:
            plt.waitforbuttonpress()
        else:
            plt.pause(0.001)

    def random_sample(self, num):
        """
        Randomly sample from the DGMM distribution
        :return tuple of sampled values and to which distribution each value
            belongs
        """
        values = []
        dists = []

        for i in range(num):
            value = normal.rvs(mean=np.zeros(self.in_dims[-1])).reshape([-1, 1])
            dist = 0

            for layer_i in range(len(self.layers) - 1, -1, -1):
                layer = self.layers[layer_i]
                dist_i = np.random.choice(len(layer),
                    p=[layer[i].pi for i in range(len(layer))])
                dist = layer[dist_i]
                value = dist.lambd @ value + dist.eta.reshape([-1, 1]) + \
                        normal.rvs(cov=dist.psi).reshape([-1, 1])
                dist = dist_i

            values.append(value.reshape(-1))
            dists.append(dist)

        return np.array(values), np.array(dists)

    @staticmethod
    def get_paths_permutations(layer_sizes):
        num = math.prod(layer_sizes)
        return np.array(
            [np.arange(num) // math.prod(layer_sizes[i + 1:]) % layer_sizes[i]
             for i in range(len(layer_sizes))]).T

    @staticmethod
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

        if AbstractDGMM.is_pd(A):
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

        while not AbstractDGMM.is_pd(A):
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
