from .AbstractDGMM import *


class CinziaDGMM(AbstractDGMM):
    SMALL_VALUE = 1e-20

    # TODO use this method instead of compute_likelihood()
    def compute_dists_prob_given_y(self, data):
        self.compute_path_distributions()
        _, prob_path_given_y = self.compute_paths_prob_given_out_values(data, 0)

        for layer_i in range(len(self.layer_sizes)):
            for dist_i in range(self.layer_sizes[layer_i]):
                # Sum over all the combinations where this distribution is
                #   present to calculate the probability that path goes through
                #   this distribution
                index = self.paths[:, layer_i] == dist_i
                dist = self.layers[layer_i][dist_i]
                dist.prob_theta_given_y = prob_path_given_y[index].sum(axis=0)

        return prob_path_given_y

    def compute_likelihood(self, data, iter_i):
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
        prob_path_given_y = prob_y_and_path / (prob_y + self.SMALL_VALUE)

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
            self.evaluate(data, iter_i)

        return prob_y, prob_y_given_path, prob_y_and_path, prob_path_given_y

    def fit(self, data):
        inv = np.linalg.pinv
        num_samples = len(data)
        self.out_dims[0] = data.shape[1]

        self._init_params(data)

        for iter_i in range(self.num_iter):
            prob_y, prob_y_given_path, prob_y_and_path, prob_path_given_y =\
                self.compute_likelihood(data, iter_i)

            # Initialize the variables
            # Shape [num_samples, dim[layer_i - 1]]
            values = data

            # Update all the layers one by one
            for layer_i in range(self.num_layers):
                layer = self.layers[layer_i]
                dim = self.in_dims[layer_i]

                # The combinations of lower layers are not included
                layer_paths_num = math.prod(self.layer_sizes[layer_i:])
                layer_paths = self.paths[:layer_paths_num]

                # Initialize variables that will be filled for each path
                # The expected values of expressions given distribution and sample y
                # E(z) at layer_i + 1, E(z.T @ z) at layer_i + 1,
                # sampled z for next layer
                exp_z_in_given_dist_z = np.zeros([len(layer), len(values), dim])
                exp_zz_in_given_dist_z = np.zeros([len(layer), len(values), dim, dim])
                dist_prob_given_value = np.zeros([len(layer), len(values)])

                for path_i in range(layer_paths_num):
                    path = layer_paths[path_i]

                    # Get the required parameters
                    dist_index = path[layer_i]
                    dist = layer[dist_index]
                    lambd, psi, eta = dist.lambd, dist.psi, dist.eta

                    if layer_i == self.num_layers - 1:
                        mu, sigma = np.zeros(dim), np.eye(dim)
                    else:
                        dist_next = self.layers[layer_i + 1][path[layer_i + 1]]
                        path_i_next = path_i % math.prod(self.layer_sizes[layer_i + 2:])
                        mu = dist_next.mu_given_path[path_i_next]
                        sigma = dist_next.sigma_given_path[path_i_next]

                    # Estimate the parameters of the p(z[k+1] | z[k]) distribution
                    ksi = inv(inv(sigma) + lambd.T @ inv(psi) @ lambd)
                    ksi = self.make_spd(ksi)

                    # Shape [num_samples, dim[l-1]]
                    rho = (ksi @ (lambd.T @ inv(psi) @ (values - eta).T
                                 + (inv(sigma) @ mu.reshape([-1, 1])))).T

                    # rho @ rho.T
                    rho_times_rho_T = np.array([r @ r.T for r in rho.reshape([*rho.shape, 1])])
                    # E(z z.T | s) = E^2(z | s) + Var(z | s)
                    expect_zz_given_path = rho_times_rho_T + ksi

                    # Sample from the distribution
                    sample_means = rho[np.random.choice(len(rho), num_samples)]
                    z_sample = normal.rvs(cov=ksi, size=num_samples)\
                        .reshape([num_samples, -1]) + sample_means

                    # Probability of the whole path given y
                    # Shape [num_samples, 1]
                    prob_path_given_y = math.prod(
                        [self.layers[i][path[i]].prob_theta_given_y
                         for i in range(layer_i + 1, len(path))]) * np.array(1)
                    prob_path_given_y = prob_path_given_y.reshape([-1, 1])

                    prob_z_given_path = normal.pdf(values, mean=mu, cov=sigma).reshape([-1])
                    dist_prob_given_value = prob_z_given_path * pi

                    exp_z_in_given_dist_z[dist_index] += rho * prob_path_given_y
                    exp_zz_in_given_dist_z[dist_index] += expect_zz_given_path * prob_path_given_y.reshape([-1, 1, 1])
                    z_times_path_prob[dist_index] += z_sample * prob_path_given_y * \
                        self.layers[layer_i][dist_index].prob_theta_given_y.reshape([-1, 1])

                # Compute the best parameter estimates for each distribution
                for dist_i in range(len(layer)):
                    dist = layer[dist_i]
                    probs = layer[dist_i].prob_theta_given_y.reshape([-1, 1])
                    denom = probs.sum() + self.SMALL_VALUE

                    exp_z = exp_z_given_dist_y[dist_i]
                    exp_zz = exp_zz_given_dist_y[dist_i]

                    normalized = values - dist.eta
                    # Shape [dim, next_dim]
                    lambd_denom = (exp_zz * probs.reshape([-1, 1, 1])) \
                        .sum(axis=0)
                    lambd = np.sum([normalized[i:i+1].T @ exp_z[i:i+1] * probs[i]
                        for i in range(len(values))], axis=0) @ inv(lambd_denom)
                    eta = ((values - (lambd @ exp_z.T).T) * probs) \
                              .sum(axis=0) / denom
                    e = values - eta - (lambd @ exp_z.T).T
                    psi = np.sum([e[i:i+1].T @ e[i:i+1] * probs[i]
                        for i in range(len(values))], axis=0) / denom
                    psi = np.diag(np.diag(psi))
                    # Make SPD. psi is diagonal, therefore it is easier
                    psi = (psi > 0) * psi + (psi <= 0) * self.SMALL_VALUE
                    pi = probs.mean()

                    dist.pi, dist.eta, dist.lambd, dist.psi = pi, eta, lambd, psi

                # Fill the values for the next layer
                # Expected value of sample
                values = z_times_path_prob.sum(0)

        # Compute the final likelihood to update the dist.prob_theta_give_y
        self.compute_likelihood(data, self.num_iter)
        return np.array([self.layers[0][dist_i].prob_theta_given_y
                         for dist_i in range(len(self.layers[0]))]).T