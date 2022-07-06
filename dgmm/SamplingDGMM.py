from .AbstractDGMM import *


class SamplingDGMM(AbstractDGMM):
    def fit(self, data):
        inv = np.linalg.pinv
        num_samples = self.num_samples
        self.out_dims[0] = data.shape[1]

        self._init_params(data)

        for iter_i in range(self.num_iter):
            self.compute_path_distributions()
            self.evaluate(data, iter_i)

            # Initialize the variables
            # Shape [num_samples, dim[layer_i - 1]]
            values = data
            values_probs = np.repeat(1, len(values))

            # Update all the layers one by one
            for layer_i in range(self.num_layers):
                layer = self.layers[layer_i]
                dim = self.in_dims[layer_i]
                dim_out = self.out_dims[layer_i]

                _, prob_paths_given_values, _, _ = \
                    self.compute_paths_prob_given_out_values(values, layer_i)

                # sampled z for next layer
                z_in_samples = []
                z_in_samples_probs = []
                pis_sum = 0

                for dist_i in range(len(layer)):
                    # The combinations of lower layers are not included
                    dist_paths_num = math.prod(self.layer_sizes[layer_i+1:])
                    dist_paths = self.paths[:dist_paths_num]
                    dist = layer[dist_i]
                    lambd, psi, eta, pi = dist.lambd, dist.psi, dist.eta, dist.pi

                    # Initialize the values for estimated parameters:
                    #   E(z), E(z @ z.T)
                    #   E(z^{+1}), E(z^{+1} @ z^{+1}.T)
                    #   E(z @ z^{+1}.T)
                    denom = 0
                    exp_z, exp_zz = np.zeros([dim_out, 1]), np.zeros([dim_out, dim_out])
                    exp_rho, exp_k = np.zeros([dim, 1]), np.zeros([dim, dim])
                    exp_z_rho = np.zeros([dim_out, dim])

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
                        ksi = self.make_spd(ksi)

                        # Shape [num_samples, dim[l-1]]
                        rho = (ksi @ (lambd.T @ inv(psi) @ (values - eta).T
                                     + (inv(sigma) @ mu.reshape([-1, 1])))).T

                        # Estimate all the variables for current path and add
                        #   up to the global estimates
                        probs = prob_paths_given_values[path_i].reshape([-1, 1]) # * values_probs
                        denom += probs.sum()
                        exp_z += (values * probs).sum(axis=0).reshape([-1, 1])
                        exp_zz += (values * probs).T @ values
                        exp_rho += (rho * probs).sum(axis=0).reshape([-1, 1])
                        exp_z_rho += (values * probs).T @ rho
                        # E(z @ z.T|s) = Var(z|s) + E^2(z|s)
                        exp_k += ksi * probs.sum() + (rho * probs).T @ rho

                    # Rescale the variables
                    exp_z /= denom
                    exp_zz /= denom
                    exp_rho /= denom
                    exp_z_rho /= denom
                    exp_k /= denom

                    # Estimate the parameters
                    lambd = (exp_z_rho - exp_z @ exp_rho.T) @ \
                            inv(exp_rho @ exp_rho.T - exp_k)
                    # TODO remove - dividing by each columns norm
                    # f = np.vectorize(lambda x: np.inf if x == 0 else x)
                    # lambd /= f(np.apply_along_axis(np.linalg.norm, 0, lambd)
                    #            .reshape([-1, 1]))
                    eta = exp_z - lambd @ exp_rho
                    psi = exp_zz - 2 * exp_z @ eta.T \
                        + eta @ eta.T + 2 * eta @ exp_rho.T @ lambd.T \
                        - 2 * exp_z_rho @ lambd.T + lambd @ exp_k @ lambd.T
                    pi = denom

                    # Make SPD. psi is diagonal, therefore it is easier
                    psi = (psi > 0) * psi + (psi <= 0) * self.SMALL_VALUE
                    # Make psi diagonal (this is a constraint we defined)
                    psi = np.diag(np.diag(psi))

                    # Set the values
                    a = 0.01

                    eta = eta.reshape([-1])
                    dist.lambd = dist.lambd * (1 - a) + lambd * a
                    dist.eta = dist.eta * (1 - a) + eta * a
                    dist.psi = dist.psi * (1 - a) + psi * a
                    dist.pi = dist.pi * (1 - a) + pi * a
                    lambd, eta, psi, pi = dist.lambd, dist.eta, dist.psi, dist.pi
                    pis_sum += pi

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
                        ksi = self.make_spd(ksi)

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

                # Rescale the pi to sum up to 1
                for dist_i in range(len(layer)):
                    layer[dist_i].pi /= pis_sum

                # Fill in the samples for next layer
                z_in_samples = np.concatenate(z_in_samples)
                z_in_samples_probs = np.concatenate(z_in_samples_probs)
                z_in_samples_probs /= z_in_samples_probs.sum()
                samples_index = np.random.choice(len(z_in_samples), num_samples)
                                                 # p=z_in_samples_probs
                values = z_in_samples[samples_index]
                values_probs = z_in_samples_probs[samples_index]

        # Compute the final distributions to update distribution parameters
        #   (mu, sigma and pi)
        self.compute_path_distributions()
        self.evaluate(data, self.num_iter)
        return self.predict(data)
