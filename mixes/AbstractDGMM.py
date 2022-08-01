import numpy as np
from scipy.stats import multivariate_normal as normal
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches
import matplotlib
from .utils import *


class AbstractDGMM:
    def __init__(self, layer_sizes, dims, plot_predictions=False,
                 plot_wait_for_input=False,
                 init='kmeans', num_iter=10, num_samples=500,
                 use_annealing=False, annealing_start_v=0.1,
                 stopping_thresh=1e-5, update_rate=0.1,
                 evaluator=None):
        def init_layer(layer_size, dim):
            tau = 1 / layer_size
            return [GaussianDistrib(dim, tau) for _ in range(layer_size)]

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

        self.paths = get_paths_permutations(self.layer_sizes)

        # Display and computation parameters
        self.plot_wait_for_input = plot_wait_for_input
        self.plot_predictions = int(plot_predictions)
        if self.plot_predictions:
            matplotlib.use("TkAgg")
            plt.ion()
            self.fig, self.ax = plt.subplots(2, 1)
            self.ax[0].set_title("Predictions plot")
            self.ax[0].set_title("Distributions plot")
            plt.draw()
            plt.show(block=False)

        self.init = init

        self.use_annealing = use_annealing
        self.annealing_v = annealing_start_v if use_annealing else 1
        self.annealing_step = (1 - self.annealing_v) / self.num_iter / 0.9

        self.log_lik = []
        self.stopping_thresh = stopping_thresh

        self.update_rate = update_rate
        self.evaluator = evaluator

    def fit(self, data):
        raise NotImplementedError(
            "Fit is not implemented in the abstract class")

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
                    dist_pi.append(dist.tau * pi[prev_path])
                    dist_mu.append(dist.eta + dist.lambd @ mu[prev_path])
                    sigma_i = dist.psi + dist.lambd @ sigma[
                        prev_path] @ dist.lambd.T
                    dist_sigma.append(make_spd(sigma_i))

                # Save the values inside distribution
                dist.pi_given_path = dist_pi
                dist.mu_given_path = dist_mu
                dist.sigma_given_path = dist_sigma
                # Append to all the mus and sigmas (at current layer)
                new_pi += dist_pi
                new_mu += dist_mu
                new_sigma += dist_sigma

            mu, sigma, pi = new_mu, new_sigma, new_pi

        return np.array(mu), np.array(sigma), np.array(pi)

    def compute_paths_prob_given_out_values(self, values, layer_i: int,
                                            annealing_value: float=1):
        """
        At a specific layer compute the probability of output values given the
        current distribution parameters
        Note that compute_path_distributions needs to be called to update
        the values of mu, sigma and pi for each distribution

        :param values: The out values
        :param layer_i: The layer number staring from 0 to self.num_layers - 1
        :param annealing_value: Value for deterministic annealing. Set to 1
            to not use annealing. Set to value from 0 to 1 for annealing
            in an optimization step. Values closer to zero will flatten the
            log-likelihood function more.
            Annealing will effectively take each probability to the power: p^v
        :return: prob_v, prob_path_given_v, prob_v_and_path
            probability of each value, probability for each path given value,
            (array of shape [len(paths), len(values)]) and probability of path
            and value (array of shape [len(paths), len(values)]
        """
        layer = self.layers[layer_i]

        # Initialize the arrays to store values in
        log_prob_v_and_path = []
        # log_prob_v_given_path = []
        for dist_i in range(len(layer)):
            # Get the values from the distribution
            mu = layer[dist_i].mu_given_path
            sigma = layer[dist_i].sigma_given_path
            pi = layer[dist_i].pi_given_path

            for path_i in range(len(mu)):
                p = normal.logpdf(values, mean=mu[path_i],
                                  cov=sigma[path_i], allow_singular=True)
                # log_prob_v_given_path.append(p)
                log_prob_v_and_path.append(p + np.log(pi[path_i]))

        # Size [n_paths, len(values)]
        # log_prob_v_given_path = np.array(log_prob_v_given_path)
        log_prob_v_and_path = np.array(log_prob_v_and_path)

        if annealing_value != 1:
            log_prob_v_and_path *= annealing_value
            # log_prob_v_given_path *= annealing_value

        # Rescale the variables for numerical stability
        log_prob_v_and_path_max = np.max(log_prob_v_and_path, axis=0)
        log_prob_v_and_path -= log_prob_v_and_path_max
        prob_v_and_path = np.exp(log_prob_v_and_path)
        prob_v = np.sum(prob_v_and_path, axis=0)
        # Use the Bayes formula p(path|v) = p(v,path) / p(v)
        prob_path_given_v = prob_v_and_path / (prob_v + SMALL_VALUE)
        prob_v *= np.exp(log_prob_v_and_path_max)

        return prob_v, prob_path_given_v, prob_v_and_path

    def predict_path_probs(self, data, annealing_value=1):
        prob_v, prob_path_given_v, _ = \
            self.compute_paths_prob_given_out_values(data, 0, annealing_value)
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
                    dist.tau = 1 / self.layer_sizes[layer_i]
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
                    dist.tau = 1 / self.layer_sizes[layer_i]
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
                    dist.tau = len(dist_values) / len(values)

                    values[kmeans.labels_ == labels[dist_i]] = (
                            np.linalg.pinv(dist.lambd) @
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
                        dist.lambd = np.repeat(
                            dist.lambd, fa.n_components, axis=1)
                    dist.psi = np.diag(fa.noise_variance_)
                    dist.tau = 1 / self.layer_sizes[layer_i]

                values = next_values
        else:
            raise ValueError("Initialization '%s' is not supported" % self.init)

    def evaluate(self, data, iter_i):
        """
        :param data:
        :param iter_i:
        :return: whether stopping criterion was reached
        """
        _, probs, prob_v = self.predict_path_probs(data)
        clusters = np.argmax(probs, 0)
        log_lik = np.sum(np.log(prob_v))

        if self.evaluator is not None:
            self.evaluator(iter_i, probs.T, clusters, log_lik)

        self.log_lik.append(log_lik)
        stopping_criterion_reached = was_stopping_criterion_reached(
            self.log_lik, self.stopping_thresh)

        if (not stopping_criterion_reached or
            not self.plot_predictions or
            iter_i % self.plot_predictions != 0):
            return False

        def draw_distribution(mean, cov, ax, color):
            # How many sigmas to draw. 2 sigmas is >95%
            N_SIGMA = 2
            mean, cov = mean[:2], cov[:2,:2]
            # Since covariance is SPD, svd will produce orthogonal eigenvectors
            U, S, V = np.linalg.svd(cov)
            # Calculate the angle of first eigenvector
            angle = float(np.degrees(np.arctan2(U[1, 0], U[0, 0])))

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

        if self.plot_wait_for_input:
            plt.waitforbuttonpress()
        else:
            plt.pause(0.001)

        return stopping_criterion_reached

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
                                          p=[layer[i].tau for i in range(len(layer))])
                dist = layer[dist_i]
                value = dist.lambd @ value + dist.eta.reshape([-1, 1]) + \
                        normal.rvs(cov=dist.psi).reshape([-1, 1])
                dist = dist_i

            values.append(value.reshape(-1))
            dists.append(dist)

        return np.array(values), np.array(dists)
