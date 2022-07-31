from .AbstractDGMM import *


class SamplingDGMM(AbstractDGMM):
    def fit(self, data):
        inv = np.linalg.pinv
        num_samples = self.num_samples
        self.out_dims[0] = data.shape[1]

        self._init_params(data)
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
                    self.compute_paths_prob_given_out_values(values, layer_i,
                        annealing_value=self.annealing_v)

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

                    # As in the paper we use
                    #   v = z[layer_i]
                    #   w = z[layer_i + 1]
                    # Initialize the values for estimated parameters:
                    #   E(v), E(v @ v.T), E(w), E(w @ w.T), E(v @ w.T)
                    denom = 0
                    exp_v, exp_vv = np.zeros([dim_out, 1]), np.zeros([dim_out, dim_out])
                    exp_w, exp_ww = np.zeros([dim, 1]), np.zeros([dim, dim])
                    exp_vw = np.zeros([dim_out, dim])

                    # TODO remove
                    # log_lik_i = 0

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
                        exp_v += (values * probs).sum(axis=0).reshape([-1, 1])
                        exp_vv += (values * probs).T @ values
                        exp_w += (rho * probs).sum(axis=0).reshape([-1, 1])
                        exp_vw += (values * probs).T @ rho
                        # E(z @ z.T|s) = Var(z|s) + E^2(z|s)
                        exp_ww += ksi * probs.sum() + (rho * probs).T @ rho

                        # TODO remove
                        # v = values - eta - (lambd @ rho.T).T
                        # log_lik_i += -0.5 * (
                        #         np.log(2 * np.pi) * probs.sum() +
                        #         np.log(np.linalg.det(psi)) * probs.sum() +
                        #         np.sum([v[i:i + 1] @ np.linalg.pinv(psi) @ v[
                        #                                                    i:i + 1].T *
                        #                 probs[i] for i in range(len(v))])
                        # )

                    # Rescale the variables
                    exp_v /= denom
                    exp_vv /= denom
                    exp_w /= denom
                    exp_vw /= denom
                    exp_ww /= denom

                    # Estimate the parameters
                    lambd = (exp_vw - exp_v @ exp_w.T) @ \
                            inv(exp_w @ exp_w.T - exp_ww)
                    eta = exp_v - lambd @ exp_w
                    # psi = exp_vv - 2 * exp_v @ eta.T \
                    #     + eta @ eta.T + 2 * eta @ exp_w.T @ lambd.T \
                    #     - 2 * exp_vw @ lambd.T + lambd @ exp_ww @ lambd.T
                    psi = exp_vv - 2 * exp_vw @ lambd.T + \
                          lambd @ exp_ww @ lambd.T - eta @ eta.T
                    pi = denom

                    # Reshape eta to its original form
                    eta = eta.reshape([-1])

                    # Make SPD. psi is diagonal, therefore it is easier
                    psi = (psi > 0) * psi + (psi <= 0) * self.SMALL_VALUE
                    # Make psi diagonal (this is a constraint we defined)
                    psi = np.diag(np.diag(psi))

                    # Perform the update
                    rate = self.update_rate
                    lambd = dist.lambd * (1 - rate) + lambd * rate
                    eta = dist.eta * (1 - rate) + eta * rate
                    psi = dist.psi * (1 - rate) + psi * rate
                    pi = dist.pi * (1 - rate) + pi * rate

                    # Set the values
                    dist.lambd, dist.eta, dist.psi, dist.pi =\
                            lambd, eta, psi, pi
                    pis_sum += pi

                    # TODO remove
                    # log_lik_i = 0

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

                        # TODO remove
                    #     v = values - eta - (lambd @ rho.T).T
                    #     log_lik_i += -0.5 * (
                    #             np.log(2 * np.pi) * probs.sum() +
                    #             np.log(np.linalg.det(psi)) * probs.sum() +
                    #             np.sum([v[i:i + 1] @ np.linalg.pinv(psi) @ v[i:i + 1].T * probs[i] for i in range(len(v))])
                    #     )
                    #
                    # # TODO remove
                    # print("  Log lik %d:%d = %.5f" % (layer_i, dist_i, log_lik_i))

                # Rescale the pi to sum up to 1
                for dist_i in range(len(layer)):
                    layer[dist_i].pi /= pis_sum

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
                self.annealing_v = np.clip(self.annealing_v, 0, 1)

            self.compute_path_distributions()
            stopping_criterion_reached = self.evaluate(data, iter_i + 1)
            if stopping_criterion_reached:
                break

        return self.predict(data)
