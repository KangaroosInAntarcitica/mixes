import numpy as np
import math
from scipy.stats import multivariate_normal as normal
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches
import matplotlib
matplotlib.use("TkAgg")


class GaussianDistrib:
    def __init__(self, pi=None, eta=None, lambd=None, psi=None):
        if pi is None:
            pi = 1

        # Parameters
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
                 init='kmeans', num_iter=10):
        def init_layer(layer_size, dim):
            pi = 1 / layer_size
            return [GaussianDistrib(dim, pi) for _ in range(layer_size)]

        self.dims = dims
        # Dimenisions on the next layer
        self.next_dims = dims[1:] + dims[-1:]
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.layers = [init_layer(layer_sizes[i], dims[i])
                       for i in range(len(dims))]
        self.num_iter = num_iter

        self.paths = self.get_paths_permutations(self.layer_sizes)

        # Display and computation parameters
        self.plot_predictions = plot_predictions
        if self.plot_predictions:
            plt.ion()
            self.fig, self.ax = plt.subplots(2, 1)
            self.ax[0].set_title("Predictions plot")
            self.ax[0].set_title("Distributions plot")
            plt.draw()
            plt.show(block=False)
        self.init = init

    def plot_prediction(self, data, iter_i):
        def draw_distribution(mean, cov, ax, color):
            # How many sigmas to draw. 2 sigmas is >95%
            N_SIGMA = 2
            mean, cov = mean[:2], cov[:2,:2]
            # Since covariance is SPD, svd will produce orthogonal eigenvectors
            U, S, V = np.linalg.svd(cov)
            # Calculate the angle of first eigenvector
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))

            # Eigenvalues are now width and height
            ax.add_patch(matplotlib.patches.Ellipse(mean, S[0] * N_SIGMA,
                S[1] * N_SIGMA, angle, color=[*color[:3], 0.3], linewidth=0))

        probs = np.array([self.layers[0][dist_i].prob_theta_given_y
                         for dist_i in range(len(self.layers[0]))]).T
        sample, sample_clust = self.random_sample(100)

        colors = cm.rainbow(np.linspace(0, 1, probs.shape[1]))

        # Draw the predictions plot
        self.ax[0].clear()
        data_colors = np.clip(probs @ colors, 0, 1)
        self.ax[0].scatter(data[:, 0], data[:, 1], color=data_colors)
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
        self.ax[1].set_aspect('equal', 'box')
        self.ax[1].set_title("Sample")
        self.ax[1].legend()

        # Draw the plots
        self.fig.suptitle("Iteration %d" % iter_i)
        plt.draw()
        plt.pause(0.001)
        # plt.waitforbuttonpress()

    def compute_likelihood(self, data, iter_i):
        # The variables pi, sigma, mu will hold the values for each path
        #   starting from the last layer to a given layer. There is only one
        #   path at the last layer - standard normal distribution
        pi = [1]
        sigma = [np.eye(self.dims[-1])]
        mu = [np.zeros(self.dims[-1])]

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
                    sigma_i = dist.psi + dist.lambd @ sigma[prev_path] @ dist.lambd.T
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

        # Initialize and fill the variables
        prob_y_given_path = []
        prob_y_and_path = []
        for path_i in range(len(self.paths)):
            p_y_path = normal.pdf(data, mean=mu[path_i], cov=sigma[path_i], allow_singular=True)
            prob_y_given_path.append(p_y_path)
            prob_y_and_path.append(pi[path_i] * p_y_path)

        # WTF? why the original code rescaled 2 times
        # prob_y_and_path /= np.max(prob_y_and_path, axis=0, keepdims=True)

        # dims = [paths, num samples] for all below
        prob_y_given_path = np.array(prob_y_given_path)
        prob_y_and_path = np.array(prob_y_and_path)
        prob_y = np.sum(prob_y_and_path, axis=0)
        prob_path_given_y = (prob_y_and_path / prob_y)

        for layer_i in range(len(self.layer_sizes)):
            for dist_i in range(self.layer_sizes[layer_i]):
                # Sum over all the combinations where this distribution is
                #   present to calculate the probability that path goes through
                #   this distribution
                index = self.paths[:, layer_i] == dist_i
                dist = self.layers[layer_i][dist_i]
                dist.prob_theta_given_y = prob_path_given_y[index].sum(axis=0)

        # s - most likelihood path for y
        # ps.y.list[layer, dist] <- prob_theta_given_y
        # hard.ps.y.list <- sampled

        if self.plot_predictions:
            self.plot_prediction(data, iter_i)

        return prob_y, prob_y_given_path, prob_y_and_path, prob_path_given_y

    def _init_params(self, data):
        if self.init == 'random':
            # Initialization
            for layer_i in range(self.num_layers):
                for dist_i in range(self.layer_sizes[layer_i]):
                    dist = self.layers[layer_i][dist_i]
                    dim, dim_next = self.dims[layer_i], self.next_dims[layer_i]

                    if layer_i == 0:
                        dist.eta = data[np.random.choice(len(data))]
                    else:
                        dist.eta = normal.rvs(mean=np.zeros(dim), cov=0.1)
                    dist.psi = np.eye(dim) / self.num_layers ** 2
                    dist.lambd = np.random.random([dim, dim_next]) / self.num_layers ** 2
                    dist.lambd /= dist.lambd.sum(axis=1, keepdims=True)
                    dist.pi = 1 / self.layer_sizes[layer_i]
            return

        if self.init == 'kmeans':
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.layer_sizes[0])
            kmeans.fit_predict(data)

            for layer_i in range(self.num_layers):
                for dist_i in range(self.layer_sizes[layer_i]):
                    dist = self.layers[layer_i][dist_i]
                    dim, dim_next = self.dims[layer_i], self.next_dims[layer_i]

                    if layer_i == 0:
                        dist.eta = kmeans.cluster_centers_[dist_i]
                    else:
                        dist.eta = normal.rvs(mean=np.zeros(dim), cov=1)
                    dist.psi = np.eye(dim) / self.num_layers ** 2
                    dist.lambd = -1 + 2 * np.random.random([dim, dim_next])
                    dist.lambd /= np.apply_along_axis(
                        np.linalg.norm, 1, dist.lambd).reshape([-1, 1])
                    dist.pi = 1 / self.layer_sizes[layer_i]
            return

        if self.init == 'kmeans':
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=len(self.paths), max_iter=200, n_init=30)
            paths_i = kmeans.fit_predict(data)
        else:
            raise ValueError("Initialization '%s' is not supported" % self.init)

        from sklearn.decomposition import FactorAnalysis

        for layer_i in range(self.num_layers):
            next_data = np.zeros([len(data), self.dims[layer_i]])

            for dist_i in range(self.layer_sizes[layer_i]):
                index = self.paths[paths_i][:, layer_i] == dist_i
                values = data[index]
                fa = FactorAnalysis(n_components=self.dims[layer_i],
                                    rotation='varimax')
                fa.fit(values)

                dist = self.layers[layer_i][dist_i]
                dist.eta = fa.mean_
                dist.lambd = fa.components_.T
                dist.psi = np.diag(fa.noise_variance_)
                dist.pi = 1 / self.layer_sizes[layer_i]

                next_data[index] = fa.transform(values)

            data = next_data

        # return np.arange(len(self.paths)) \
        #            .repeat(math.ceil(len(data) / len(self.paths)))[:len(data)]

    def fit(self, data):
        inv = np.linalg.pinv
        num_observations = len(data)

        self._init_params(data)

        for iter_i in range(self.num_iter):
            prob_y, prob_y_given_path, prob_y_and_path, prob_path_given_y =\
                self.compute_likelihood(data, iter_i)

            # Initialize the variables
            # Shape [num_observations, dim[layer_i - 1]]
            values = data

            # Update all the layers one by one
            for layer_i in range(self.num_layers):
                layer = self.layers[layer_i]
                dim = self.next_dims[layer_i]

                # The combinations of lower layers are not included
                layer_paths_num = math.prod(self.layer_sizes[layer_i:])
                layer_paths = self.paths[:layer_paths_num]

                # Initialize variables that will be filled for each path
                # The expected values of expressions given distribution and sample y
                # E(z) at layer_i + 1, E(z.T @ z) at layer_i + 1,
                # sampled z for next layer
                exp_z_given_dist_y = np.zeros([len(layer), num_observations, dim])
                exp_zz_given_dist_y = np.zeros([len(layer), num_observations, dim, dim])
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
                    rho = (ksi @ (lambd.T @ inv(psi) @ (values - eta).T
                                 + (inv(sigma) @ mu.reshape([-1, 1])))).T

                    # rho @ rho.T
                    rho_times_rho_T = np.array([r @ r.T for r in rho.reshape([*rho.shape, 1])])
                    # E(z z.T | s) = E^2(z | s) + Var(z | s)
                    expect_zz_given_path = rho_times_rho_T + ksi

                    # Sample from the distribution
                    z_sample = normal.rvs(cov=ksi, size=num_observations)\
                                   .reshape([num_observations, -1]) + rho

                    # Probability of the whole path given y
                    # Shape [num_observations, 1]
                    prob_path_given_y = math.prod(
                        [self.layers[i][path[i]].prob_theta_given_y
                         for i in range(layer_i + 1, len(path))]) * np.array(1)
                    prob_path_given_y = prob_path_given_y.reshape([-1, 1])

                    exp_z_given_dist_y[dist_index] += rho * prob_path_given_y
                    exp_zz_given_dist_y[dist_index] += expect_zz_given_path * prob_path_given_y.reshape([-1, 1, 1])
                    z_times_path_prob[dist_index] += z_sample * prob_path_given_y

                # Compute the best parameter estimates for each distribution
                for dist_i in range(len(layer)):
                    dist = layer[dist_i]
                    exp_z = exp_z_given_dist_y[dist_i]
                    exp_zz = exp_zz_given_dist_y[dist_i]

                    probs = layer[dist_i].prob_theta_given_y.reshape([-1, 1])
                    denom = probs.sum()
                    normalized = values - dist.eta
                    # Shape [dim, next_dim]
                    lambd = np.sum([normalized[i:i+1].T @
                        exp_z[i:i+1] @ inv(exp_zz[i]) * probs[i]
                        for i in range(num_observations)], axis=0) / denom
                    psi = np.sum([(normalized[i:i+1].T @ normalized[i:i+1] -
                        normalized[i:i+1].T @ exp_z[i:i+1] @ lambd.T) * probs[i]
                        for i in range(num_observations)], axis=0) / denom
                    psi = self.make_spd(psi)
                    eta = ((values - (lambd @ exp_z.T).T) * probs)\
                              .sum(axis=0) / denom
                    pi = probs.mean()

                    dist.pi, dist.eta, dist.lambd, dist.psi = pi, eta, lambd, psi

                # Fill the values for the next layer
                # Expected value of sample
                values = z_times_path_prob.sum(0)

        # Compute the final likelihood to update the dist.prob_theta_give_y
        self.compute_likelihood(data, self.num_iter)
        return np.array([self.layers[0][dist_i].prob_theta_given_y
                         for dist_i in range(len(self.layers[0]))]).T

    def random_sample(self, num):
        """
        Randomly sample from the DGMM distribution
        :return tuple of sampled values and to which distribution each value
            belongs
        """
        values = []
        dists = []

        for i in range(num):
            value = normal.rvs(mean=np.zeros(self.dims[-1])).T * np.array([1])
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

