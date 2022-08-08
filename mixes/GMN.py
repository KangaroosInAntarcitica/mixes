from scipy.stats import multivariate_normal as normal
from .utils import *


class GMN:
    def __init__(self, layer_sizes, dims,
                 init='kmeans', num_iter=100, num_samples=500,
                 use_annealing=False, annealing_start_v=0.1, update_rate=1,
                 evaluator=None, hard_distribution=False,
                 stopping_criterion=None,
                 var_regularization=1e-4):
        def init_layer(layer_size, next_layer_size):
            tau = np.ones([next_layer_size]) / layer_size
            return [GaussianDistrib(tau.copy()) for _ in range(layer_size)]

        if len(layer_sizes) != len(dims):
            raise ValueError("The size of layer_sizes and dims must match")

        # Dimensions on the input (next layer)
        self.in_dims = dims
        # Dimensions on the output (prev layer)
        # First value will be the dimensions of the data
        self.out_dims = [0] + dims[:-1]

        self.layer_sizes = layer_sizes
        self.next_layer_sizes = layer_sizes[1:] + [1]
        self.num_layers = len(layer_sizes)
        self.layers = [init_layer(layer_sizes[i], self.next_layer_sizes[i])
                       for i in range(len(dims))]
        self.num_iter = num_iter
        self.num_samples = num_samples

        self.paths = get_paths_permutations(self.layer_sizes)

        self.init = init
        self.was_init = False

        self.use_annealing = use_annealing
        self.annealing_v = annealing_start_v if use_annealing else 1
        self.annealing_step = (1 - self.annealing_v) /\
                              (self.num_iter + SMALL_VALUE) / 0.9
        self.hard_distribution = hard_distribution

        self.stopping_criterion = stopping_criterion

        self.update_rate = update_rate
        self.evaluator = evaluator
        self.var_regularization = var_regularization

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
                    prev_dist = prev_path // \
                                int(len(sigma) / self.next_layer_sizes[layer_i])

                    # Compute new pi, mu and sigma (also make it spd)
                    dist_pi.append(dist.tau[prev_dist] * pi[prev_path])
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
                                            annealing_value: float = 1,
                                            hard_distributions=False):
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
        :param hard_distributions:
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
                log_prob_v_and_path.append(p + np.log(pi[path_i] + SMALL_VALUE))

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

        if hard_distributions:
            max_index = np.argmax(prob_path_given_v, 0)
            prob_path_given_v_hard = np.zeros_like(prob_path_given_v)
            prob_path_given_v_hard[max_index, np.arange(len(prob_v))] = 1

            prob_path_given_v *= 0.1
            prob_path_given_v += prob_path_given_v_hard * 0.9

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
        return prob_dist_given_v.T if probs else \
            np.argmax(prob_dist_given_v, 0)

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
                    dist.lambd = np.random.random([dim, in_dim]) \
                                 / self.num_layers ** 2
                    dist.lambd /= dist.lambd.sum(axis=0, keepdims=True)
                    dist.tau = np.ones_like(dist.tau) / self.layer_sizes[layer_i]
        elif self.init == 'kmeans':
            from sklearn.cluster import KMeans
            from sklearn.decomposition import FactorAnalysis
            import warnings

            values = data
            prev_clusters = None
            prev_cluster_i = None

            for layer_i in range(self.num_layers):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kmeans = KMeans(n_clusters=self.layer_sizes[layer_i],
                                    n_init=30, max_iter=200)
                    clusters = kmeans.fit_predict(values)
                cluster_i = np.unique(clusters)
                next_values = np.zeros([len(values), self.in_dims[layer_i]])

                for dist_i in range(self.layer_sizes[layer_i]):
                    if len(cluster_i) <= dist_i:
                        dist = self.layers[layer_i][dist_i]
                        prev_dist = self.layers[layer_i][dist_i - 1]
                        dist.eta = prev_dist.eta
                        dist.lambd = prev_dist.lambd
                        dist.psi = prev_dist.psi
                        dist.tau = prev_dist.tau = prev_dist.tau / 2
                        continue

                    index = clusters == cluster_i[dist_i]
                    values_i = values[index]
                    fa = FactorAnalysis(n_components=self.in_dims[layer_i],
                                        rotation='varimax')
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        next_values_i = fa.fit_transform(values_i)
                    # next_values[index] = (np.linalg.pinv(fa.components_.T) @
                    #                       (values_i - fa.mean_).T).T
                    if next_values_i.shape[1] < self.in_dims[layer_i]:
                        dims_r = self.in_dims[layer_i] - next_values_i.shape[1]
                        values_r = np.zeros([len(next_values_i), dims_r])
                        next_values_i = np.concatenate([next_values_i, values_r], 1)
                    next_values[index] = next_values_i

                    dist = self.layers[layer_i][dist_i]
                    dist.eta = fa.mean_

                    dist.lambd = fa.components_.T
                    if dist.lambd.shape[1] < self.in_dims[layer_i]:
                        dims_r = self.in_dims[layer_i] - dist.lambd.shape[1]
                        lambd_r = np.zeros([len(dist.lambd), dims_r])
                        dist.lambd = np.concatenate([dist.lambd, lambd_r], 1)
                    dist.psi = np.diag(fa.noise_variance_)

                    if layer_i != 0:
                        tau_sum = 0
                        for prev_dist_i in range(self.layer_sizes[layer_i - 1]):
                            prev_index = prev_clusters == prev_cluster_i[prev_dist_i]
                            prob = index[prev_index].sum() / index.sum() + SMALL_VALUE
                            tau_sum += prob
                            self.layers[layer_i - 1][prev_dist_i].tau[dist_i] = prob

                        for prev_dist_i in range(self.layer_sizes[layer_i - 1]):
                            self.layers[layer_i - 1][prev_dist_i].tau[dist_i] /= tau_sum

                if layer_i != 0:
                    for prev_dist_i in range(self.layer_sizes[layer_i - 1]):
                        prev_dist = self.layers[layer_i - 1][prev_dist_i]
                        prev_dist.tau = np.array(prev_dist.tau)

                values = next_values
                prev_clusters = clusters
                prev_cluster_i = cluster_i
        else:
            raise ValueError("Initialization '%s' is not supported" % self.init)

    def plot_predictions(self, data, probs=None, ax=None):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        if ax is not None:
            plt.sca(ax)
        if probs is None:
            _, probs, _ = self.predict_path_probs(data)

        colors = cm.rainbow(np.linspace(0, 1, probs.shape[0]))

        # Draw the predictions plot
        data_colors = np.clip(probs.T @ colors, 0, 1)
        plt.scatter(data[:, 0], data[:, 1], color=data_colors, s=10)
        plt.gca().set_aspect('equal', 'box')
        plt.gca().set_title("Probabilities")

    def plot_distributions(self, data, probs=None, ax=None, draw_samples=False,
                           different_path_colors=False, use_pi=False, colors=None,
                           axis=[0, 1]):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.patches
        import matplotlib

        def draw_distribution(mean, cov, pi, ax, color):
            # How many sigmas to draw. 2 sigmas is >95%
            N_SIGMA = 1
            index = [axis[0], axis[1]]
            mean, cov = mean[index], cov[index][:, index]
            # Since covariance is SPD, svd will produce orthogonal eigenvectors
            U, S, V = np.linalg.svd(cov)
            # Calculate the angle of first eigenvector
            angle = float(np.degrees(np.arctan2(U[1, 0], U[0, 0])))

            # Eigenvalues are now half-width and half-height squared
            std = np.sqrt(S)
            factor = 2 * N_SIGMA * (pi * n_paths if use_pi else 1)
            alpha = 1 / n_paths * self.layer_sizes[0]
            ellipse = matplotlib.patches.Ellipse(
                mean, std[0] * factor, std[1] * factor, angle,
                color=[*color[:3], alpha], linewidth=0)
            ax.add_patch(ellipse)

        if ax is not None:
            plt.sca(ax)
        if probs is None:
            _, probs, _ = self.predict_path_probs(data)

        n_dists = probs.shape[0]
        n_paths = len(self.paths)
        if colors is None:
            colors = cm.rainbow(np.linspace(
                0, 1, n_paths if different_path_colors else n_dists))

        # Draw the distributions plot
        for dist_i in range(self.layer_sizes[0]):
            dist = self.layers[0][dist_i]
            for path_i in range(len(dist.mu_given_path)):
                color = colors[dist_i * int(n_paths / n_dists) + path_i]\
                    if different_path_colors else colors[dist_i]
                draw_distribution(dist.mu_given_path[path_i],
                                  dist.sigma_given_path[path_i],
                                  dist.pi_given_path[path_i],
                                  plt.gca(), color)

        # Draw the samples plot
        if draw_samples:
            sample, sample_clust = self.random_sample(200)
            for dist_i in range(self.layer_sizes[0]):
                s_values = sample[sample_clust == dist_i]
                color = colors[dist_i]
                plt.gca().scatter(s_values[:, axis[0]], s_values[:, axis[1]], color=color,
                                   label="%d" % (dist_i + 1), s=10)
            plt.legend()

        plt.gca().set_aspect('equal', 'box')
        # recompute the ax.dataLim and update ax.viewLim using the new dataLim
        plt.gca().relim()
        plt.gca().autoscale_view()
        plt.title("Distributions")

    def evaluate(self, data, iter_i):
        """
        :param data:
        :param iter_i:
        :return: log-likelihood value
        """
        _, probs, prob_v = self.predict_path_probs(data)
        clusters = np.argmax(probs, 0)
        log_lik = np.sum(np.log(prob_v)) if np.all(prob_v != 0) else -np.inf

        if self.evaluator is not None:
            self.evaluator(iter_i, data, probs.T, clusters, log_lik)

        stopping_criterion_reached = \
            self.stopping_criterion(iter_i, data, probs.T, clusters, log_lik) \
            if self.stopping_criterion is not None else False

        return stopping_criterion_reached

    def random_sample(self, num, return_paths=False):
        """
        Randomly sample from the DGMM distribution
        :return tuple of sampled values and to which distribution each value
            belongs
        """
        values = []
        dists = []
        paths = []

        for i in range(num):
            value = normal.rvs(mean=np.zeros(self.in_dims[-1])).reshape([-1, 1])
            dist = 0
            path = []

            for layer_i in range(len(self.layers) - 1, -1, -1):
                layer = self.layers[layer_i]
                dist_i = np.random.choice(len(layer),
                                          p=[layer[i].tau[dist] for i in range(len(layer))])
                dist = layer[dist_i]
                value = dist.lambd @ value + dist.eta.reshape([-1, 1]) + \
                        normal.rvs(cov=dist.psi).reshape([-1, 1])
                dist = dist_i
                path.append(dist_i)

            values.append(value.reshape(-1))
            dists.append(dist)
            paths.append(path[::-1])

        if return_paths:
            return np.array(values), np.array(paths)
        return np.array(values), np.array(dists)

    def fit(self, data):
        inv = np.linalg.pinv
        num_samples = self.num_samples
        self.out_dims[0] = data.shape[1]

        if not self.was_init:
            self._init_params(data)
            self.was_init = True
        self.compute_path_distributions()
        self.evaluate(data, 0)

        for iter_i in range(self.num_iter):
            # Initialize the variables
            # Shape [num_samples, dim[layer_i - 1]]
            values = data
            values_probs = np.repeat(1, len(values))

            # Update all the layers one by one
            for layer_i in range(self.num_layers):
                layer = self.layers[layer_i]
                dim = self.in_dims[layer_i]
                dim_out = self.out_dims[layer_i]

                _, prob_paths_given_values, _ = \
                    self.compute_paths_prob_given_out_values(
                        values, layer_i,
                        annealing_value=self.annealing_v,
                        hard_distributions=self.hard_distribution)

                # sampled z for next layer
                z_in_samples = []
                z_in_samples_probs = []
                tau_sum = 0

                for dist_i in range(len(layer)):
                    # The combinations of lower layers are not included
                    dist_paths_num = math.prod(self.layer_sizes[layer_i+1:])
                    dist_paths = self.paths[:dist_paths_num]
                    dist = layer[dist_i]
                    lambd, psi, eta, tau = dist.lambd, dist.psi, dist.eta, dist.tau

                    # As in the paper we use
                    #   v = z[layer_i]
                    #   w = z[layer_i + 1]
                    # Initialize the values for estimated parameters:
                    #   E(v), E(v @ v.T), E(w), E(w @ w.T), E(v @ w.T)
                    denom = 0
                    exp_v, exp_vv = np.zeros([dim_out, 1]), np.zeros([dim_out, dim_out])
                    exp_w, exp_ww = np.zeros([dim, 1]), np.zeros([dim, dim])
                    exp_vw = np.zeros([dim_out, dim])
                    tau_probs = np.zeros_like(dist.tau)

                    for dist_path_i in range(dist_paths_num):
                        path_i = dist_path_i + dist_paths_num * dist_i
                        path = dist_paths[dist_path_i]

                        if layer_i == self.num_layers - 1:
                            mu, sigma = np.zeros(dim), np.eye(dim)
                        else:
                            dist_next = self.layers[layer_i + 1][path[layer_i+1]]
                            next_path_i = dist_path_i % math.prod(self.layer_sizes[layer_i+2:])
                            mu = dist_next.mu_given_path[next_path_i]
                            sigma = dist_next.sigma_given_path[next_path_i]

                        # Estimate the parameters of the p(z[k+1] | z[k]) distribution
                        ksi = inv(inv(sigma) + lambd.T @ inv(psi) @ lambd)
                        ksi = make_spd(ksi)

                        # Shape [num_samples, dim[l-1]]
                        rho = (ksi @ (lambd.T @ inv(psi) @ (values - eta).T
                                     + (inv(sigma) @ mu.reshape([-1, 1])))).T

                        # Estimate all the variables for current path and add
                        #   up to the global estimates
                        probs = prob_paths_given_values[path_i].reshape([-1, 1]) # * values_probs
                        denom += probs.sum()
                        exp_v += (values * probs).sum(axis=0).reshape([-1, 1])
                        exp_vv += (values * probs).T @ values
                        exp_w += (rho * probs).sum(axis=0).reshape([-1, 1])
                        exp_vw += (values * probs).T @ rho
                        # E(z @ z.T|s) = Var(z|s) + E^2(z|s)
                        exp_ww += ksi * probs.sum() + (rho * probs).T @ rho

                        if layer_i != self.num_layers - 1:
                            next_dist_i = path[layer_i + 1]
                            tau_probs[next_dist_i] += probs.sum()
                        else:
                            tau_probs += probs.sum()

                    # Rescale the variables
                    exp_v /= denom
                    exp_vv /= denom
                    exp_w /= denom
                    exp_vw /= denom
                    exp_ww /= denom

                    # Estimate the parameters
                    lambd = (exp_vw - exp_v @ exp_w.T) @ \
                            inv(exp_ww - exp_w @ exp_w.T)
                    eta = exp_v - lambd @ exp_w
                    # psi = exp_vv - 2 * exp_v @ eta.T \
                    #     + eta @ eta.T + 2 * eta @ exp_w.T @ lambd.T \
                    #     - 2 * exp_vw @ lambd.T + lambd @ exp_ww @ lambd.T
                    psi = exp_vv - 2 * exp_vw @ lambd.T \
                          + lambd @ exp_ww @ lambd.T - eta @ eta.T
                    tau = tau_probs

                    # Reshape eta to its original form
                    eta = eta.reshape([-1])

                    # Make SPD. psi is diagonal, therefore it is easier
                    psi = (psi > 0) * psi + (psi <= 0) * SMALL_VALUE
                    # Make psi diagonal (this is a constraint we defined)
                    psi = np.diag(np.diag(psi))

                    # Add regularization
                    lambd2 = lambd @ lambd.T + \
                             np.eye(len(psi)) * self.var_regularization
                    l, d, _ = np.linalg.svd(lambd2, hermitian=True)
                    i = np.argsort(d)[-lambd.shape[1]:]
                    lambd = l[:,i] @ np.diag(np.sqrt(d[i]))
                    psi = psi + np.eye(len(psi)) * self.var_regularization

                    # Perform the update
                    rate = self.update_rate
                    lambd = dist.lambd * (1 - rate) + lambd * rate
                    eta = dist.eta * (1 - rate) + eta * rate
                    psi = dist.psi * (1 - rate) + psi * rate
                    tau = dist.tau * (1 - rate) + tau * rate

                    # Set the values
                    dist.lambd, dist.eta, dist.psi, dist.tau =\
                        lambd, eta, psi, tau
                    tau_sum += tau

                    # Sample values for next layer
                    for dist_path_i in range(dist_paths_num):
                        path = dist_paths[dist_path_i]
                        path_i = dist_path_i + dist_paths_num * dist_i

                        if layer_i == self.num_layers - 1:
                            mu, sigma = np.zeros(dim), np.eye(dim)
                        else:
                            dist_next = self.layers[layer_i + 1][path[layer_i + 1]]
                            next_path_i = dist_path_i % math.prod(self.layer_sizes[layer_i + 2:])
                            mu = dist_next.mu_given_path[next_path_i]
                            sigma = dist_next.sigma_given_path[next_path_i]

                        # Estimate the parameters of the p(z[k+1] | z[k]) distribution
                        ksi = inv(inv(sigma) + lambd.T @ inv(psi) @ lambd)
                        ksi = make_spd(ksi)

                        # Shape [num_samples, dim[l-1]]
                        rho = (ksi @ (lambd.T @ inv(psi) @ (values - eta).T
                                     + (inv(sigma) @ mu.reshape([-1, 1])))).T

                        # Sample from the distribution
                        probs = prob_paths_given_values[path_i] # * values_probs
                        sample_index = np.random.choice(len(rho), num_samples)
                        sample_means = rho[sample_index]
                        sample_probs = probs[sample_index]
                        z_sample = normal.rvs(cov=ksi, size=num_samples)\
                            .reshape([num_samples, -1]) + sample_means
                        z_in_samples.append(z_sample)
                        z_in_samples_probs.append(sample_probs)

                # Rescale the tau to sum up to 1
                for dist_i in range(len(layer)):
                    layer[dist_i].tau /= tau_sum

                # Fill in the samples for next layer
                z_in_samples = np.concatenate(z_in_samples)
                z_in_samples_probs = np.concatenate(z_in_samples_probs)
                z_in_samples_probs /= z_in_samples_probs.sum()
                samples_index = np.random.choice(len(z_in_samples), num_samples,
                                                 p=z_in_samples_probs)
                values = z_in_samples[samples_index]
                values_probs = z_in_samples_probs[samples_index]

            # Finish iteration
            if self.use_annealing:
                self.annealing_v += self.annealing_step
                self.annealing_v = float(np.clip(self.annealing_v, 0, 1))

            self.compute_path_distributions()
            stopping_criterion_reached = self.evaluate(data, iter_i + 1)
            if stopping_criterion_reached:
                break

        return self.predict(data)
