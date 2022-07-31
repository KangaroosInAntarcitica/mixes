from .AbstractDGMM import *


class GradientDescentDGMM(AbstractDGMM):
    def __init__(self, layer_sizes, dims, step_size=0.1, *args, **kwargs):
        super().__init__(layer_sizes, dims, *args, **kwargs)

        self.step_size = step_size

    def fit(self, data):
        inv = np.linalg.inv
        self.out_dims[0] = data.shape[1]

        self._init_params(data)

        for iter_i in range(self.num_iter):
            mu, sigma, pi = self.compute_path_distributions()
            self.evaluate(data, iter_i)

            sample = data[np.random.choice(len(data), self.num_samples,
                                           replace=False)]
            _, prob_path_given_v, prob_v_and_path = \
                self.compute_paths_prob_given_out_values(sample, 0)

            dmu = []
            dsigma = []
            dpi = []

            # Calculate the derivative w.r.t. the path parameters
            #   (Mean is used to make the behavior more predictable)
            for path_i in range(len(self.paths)):
                mu_i, sigma_i, pi_i = mu[path_i], sigma[path_i], pi[path_i]
                sigma_inv = inv(sigma_i + np.eye(len(sigma_i)) * 0.0001)

                probs = prob_path_given_v[path_i].reshape([-1, 1])
                # The probs should not be scaled the same way for all paths
                #   therefore denom is just the number of samples
                denom = len(sample) # probs.sum() + self.SMALL_VALUE
                dmu.append((sigma_inv @ (np.sum(sample * probs, axis=0) / denom - mu_i)
                            .reshape([-1, 1])).reshape([-1]))
                dsigma.append(0.5 * (-sigma_inv +
                    np.sum([sigma_inv @ (sample[i:i+1] - mu_i).T @
                            (sample[i:i+1] - mu_i) @ sigma_inv * probs[i]
                            for i in range(len(sample))], axis=0) / denom))
                dpi.append(-0.5 / pi[path_i])

            dmu, dsigma, dpi = \
                np.array(dmu), np.array(dsigma), np.array(dpi)

            for layer_i in range(self.num_layers):
                n_dist_paths = math.prod(self.layer_sizes[layer_i + 1:])

                dim = self.in_dims[layer_i]
                dmu_new = np.zeros([n_dist_paths, dim])
                dsigma_new = np.zeros([n_dist_paths, dim, dim])
                pis_sum = 0

                for dist_i in range(self.layer_sizes[layer_i]):
                    # The paths that go through current distribution
                    paths_index = np.arange(len(dmu)) // n_dist_paths == dist_i
                    dist = self.layers[layer_i][dist_i]

                    # Calculate the derivatives
                    dpsi = np.diag(np.diag(dsigma[paths_index].sum(axis=0)))
                    deta = dmu[paths_index].sum(axis=0)
                    dlambd = 0
                    dp = (dpi[self.paths[:, layer_i] == dist_i] *
                          pi[self.paths[:, layer_i] == dist_i]).sum() / dist.pi

                    for dist_path_i in range(n_dist_paths):
                        next_dist_paths = math.prod(self.layer_sizes[layer_i + 2:])
                        if layer_i == self.num_layers - 1:
                            s = np.eye(self.in_dims[layer_i])
                        else:
                            s = self.layers[layer_i + 1]\
                                [dist_path_i // next_dist_paths]\
                                .sigma_given_path[dist_path_i % next_dist_paths]
                        right = 2 * dist.lambd @ s
                        dlambd += dsigma[paths_index].sum(axis=0) @ right

                    # Calculate the derivatives for the next layer
                    dmu_new += dmu[paths_index] @ dist.lambd
                    dsigma_new += ((dsigma[paths_index] @ dist.lambd).transpose([0, 2, 1]) @ dist.lambd)

                    # Calculate the step size
                    step_size = (self.step_size /
                        math.prod(self.layer_sizes[layer_i+1:]) /
                        math.prod(self.layer_sizes[:layer_i]))

                    sqrt_psi = np.sqrt(dist.psi)
                    d_sqrt_psi = 2 * dpsi * sqrt_psi
                    sqrt_psi += d_sqrt_psi * step_size
                    dist.psi = sqrt_psi ** 2

                    # Perform the gradient step
                    dist.lambd += dlambd * step_size
                    dist.eta += deta * step_size
                    # dist.psi += dpsi * step_size
                    # dist.psi = (dist.psi > 0) * dist.psi + \
                    #            (dist.psi <= 0) * 0.0001
                    dist.pi += dp * step_size
                    pis_sum += dist.pi

                    # dist.lambd /= np.apply_along_axis(np.linalg.norm, 0, dist.lambd)
                    print("\t %d:%d, lambd = %.3f, eta = %.3f, psi = %.3f, pi = %.3f" %
                          (layer_i ,dist_i,
                           dlambd.mean() * step_size,
                           deta.mean() * step_size,
                           d_sqrt_psi.mean() * step_size,
                           dp.mean() * step_size))

                # Rescale the pis
                for dist_i in range(self.layer_sizes[layer_i]):
                    self.layers[layer_i][dist_i].pi /= pis_sum

                dsigma, dmu = dsigma_new, dmu_new

            # End of iteration

        # Compute the final distributions to update distribution parameters
        #   (mu, sigma and pi)
        self.compute_path_distributions()
        self.evaluate(data, self.num_iter)

        return self.predict(data)
