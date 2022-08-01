from .AbstractDGMM import AbstractDGMM
import math
from scipy.stats import multivariate_normal as normal
import numpy as np
import utils


class ViroliDGMM(AbstractDGMM):
    """
    Not inteded for use. Use SamplingDGMM instead.

    An attempt of implementing DGMM based on implementation of paper
    "Deep Gaussian mixture models" by Cinzia Viroli, Geoffrey J. McLachlan (2019)
    https://link.springer.com/article/10.1007/s11222-017-9793-z

    Code for R is present in github repository:
        https://github.com/suren-rathnayake/deepgmm

    Alternative implementation is for paper
    "A bumpy journey: exploring deep gaussian mixture models" by M. Selosse et. al
    and is in github respository:
        https://github.com/ansubmissions/ICBINB
    """
    def compute_dists_prob_given_y(self, data):
        self.compute_path_distributions()
        _, prob_path_given_y, _ = self.compute_paths_prob_given_out_values(data, 0)

        for layer_i in range(len(self.layer_sizes)):
            for dist_i in range(self.layer_sizes[layer_i]):
                # Sum over all the combinations where this distribution is
                #   present to calculate the probability that path goes through
                #   this distribution
                index = self.paths[:, layer_i] == dist_i
                dist = self.layers[layer_i][dist_i]
                dist.prob_theta_given_y = prob_path_given_y[index].sum(axis=0)

        return prob_path_given_y

    def fit(self, data):
        def inv(M):
            u, s, vh = np.linalg.svd(M)
            pos = s > max(1e-20 * s[0], 0)
            if np.all(pos):
                return vh @ (1 / s * u).T
            elif not np.any(pos):
                return np.zeros_like(M.T)
            else:
                return vh[:, pos] @ (1 / s[pos] * u[:, pos]).T

        inv = np.linalg.pinv

        num_samples = len(data)
        self.out_dims[0] = data.shape[1]

        self._init_params(data)

        for iter_i in range(self.num_iter):
            self.compute_dists_prob_given_y(data)
            self.evaluate(data, iter_i)

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
                exp_sample_given_z = np.zeros([len(values), dim])

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
                    psi_inv = np.diag(1 / np.diag(psi))
                    ksi = inv(inv(sigma) + lambd.T @ psi_inv @ lambd)
                    # Shape [num_samples, dim[l-1]]
                    rho = (ksi @ (lambd.T @ psi_inv @ (values - eta).T
                                 + (inv(sigma) @ mu.reshape([-1, 1])))).T
                    # ksi = self.make_spd(ksi)

                    # rho @ rho.T
                    rho_times_rho_T = np.array([r @ r.T for r in rho.reshape([*rho.shape, 1])])
                    # E(z z.T | s) = E^2(z | s) + Var(z | s)
                    expect_zz_given_path = rho_times_rho_T + ksi

                    # Sample from the distribution
                    z_sample = normal.rvs(cov=ksi, size=num_samples)\
                        .reshape([num_samples, -1]) + rho

                    # Probability of the whole path given y
                    # Shape [num_samples, 1]
                    prob_path_given_y = math.prod(
                        [self.layers[i][path[i]].prob_theta_given_y
                         for i in range(layer_i + 1, len(path))]) * np.array(1)
                    prob_path_given_y = prob_path_given_y.reshape([-1, 1])

                    exp_z_in_given_dist_z[dist_index] += rho * prob_path_given_y
                    exp_zz_in_given_dist_z[dist_index] += expect_zz_given_path * prob_path_given_y.reshape([-1, 1, 1])
                    exp_sample_given_z += z_sample * prob_path_given_y * \
                        self.layers[layer_i][dist_index].prob_theta_given_y.reshape([-1, 1])

                pis_sum = 0

                # Compute the best parameter estimates for each distribution
                for dist_i in range(len(layer)):
                    dist = layer[dist_i]
                    probs = dist.prob_theta_given_y.reshape([-1, 1])
                    denom = probs.sum() + SMALL_VALUE

                    exp_z = exp_z_in_given_dist_z[dist_i]
                    exp_zz = exp_zz_in_given_dist_z[dist_i]

                    exp_zz = np.sum(exp_zz * probs.reshape([-1, 1, 1]), axis=0) / denom

                    normalized = (values - dist.eta) * probs
                    # Shape [dim, next_dim]
                    lambd = np.sum([normalized[i:i+1].T @ (exp_z[i:i+1] * probs[i])
                                    for i in range(len(values))], axis=0) @ inv(exp_zz) / denom
                    psi = np.sum([normalized[i:i+1].T @ normalized[i:i+1] -
                                  normalized[i:i+1].T @ (exp_z[i:i+1] * probs[i]) @ lambd.T
                                  for i in range(len(values))], axis=0) / denom
                    psi = np.diag(np.diag(psi))
                    # Make SPD. psi is diagonal, therefore it is easier
                    psi = (psi > 0) * psi + (psi <= 0) * 0.000001
                    eta = ((values - (lambd @ exp_z.T).T) * probs).sum(axis=0) / denom
                    pi = probs.mean() + SMALL_VALUE
                    pis_sum += pi

                    print("%d:%d, eta=%.3g, lambd=%.3g, psi=%.3g" %
                          (layer_i, dist_i, eta.mean(), lambd.mean(), psi.mean()))
                    dist.tau, dist.eta, dist.lambd, dist.psi = pi, eta, lambd, psi

                # Rescale the pis
                for dist_i in range(len(layer)):
                    layer[dist_i].tau /= pis_sum

                # Fill the values for the next layer
                # Expected value of sample
                values = exp_sample_given_z

        # Compute the final likelihood to update the dist.prob_theta_give_y
        self.compute_dists_prob_given_y(data)
        self.evaluate(data, self.num_iter)
        return self.predict(data)

    def compute_likelihood(self, data):
        mu, sigma, pi = self.compute_path_distributions()

        # Initialize and fill the variables
        log_prob_y_given_path = []
        log_prob_y_and_path = []
        for path_i in range(len(self.paths)):
            log_p_y_path = normal.logpdf(
                data, mean=mu[path_i], cov=sigma[path_i], allow_singular=True)
            log_prob_y_given_path.append(log_p_y_path)
            log_prob_y_and_path.append(np.log(pi[path_i]) + log_p_y_path)

        # Size [n_paths, len(data)]
        log_prob_y_and_path = np.array(log_prob_y_and_path)
        log_prob_y_given_path = np.array(log_prob_y_given_path)

        # Rescale the variables for numerical stability
        log_prob_y_and_path_max = np.max(log_prob_y_and_path, axis=0)
        log_prob_y_and_path -= log_prob_y_and_path_max
        prob_y_and_path = np.exp(log_prob_y_and_path)
        prob_y = np.sum(prob_y_and_path, axis=0)
        prob_path_given_y = prob_y_and_path / prob_y
        prob_y *= np.exp(log_prob_y_and_path_max)

        for layer_i in range(len(self.layer_sizes)):
            for dist_i in range(self.layer_sizes[layer_i]):
                # Sum over all the combinations where this distribution is
                #   present to calculate the probability that path goes through
                #   this distribution
                index = self.paths[:, layer_i] == dist_i
                dist = self.layers[layer_i][dist_i]
                dist.prob_theta_given_y = prob_path_given_y[index].sum(axis=0)

        return prob_y, prob_y_and_path, prob_path_given_y
