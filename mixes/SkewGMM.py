from scipy.stats import multivariate_normal as normal
from scipy.stats import mvn
from .utils import *
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


class SkewGMM:
    # Source: "Maximum likelihood estimation for multivariate skew normal
    # mixture models" Tsung I. Lin (2006)
    # https://www.sciencedirect.com/science/article/pii/S0047259X08001152
    # R code:
    # https://github.com/cran/mixsmsn/blob/master/R/smsn.mmix.R

    def __init__(self, num_dists, num_iter=10, evaluator=None, update_rate=0.1,
                 use_annealing=False, annealing_start_v=0.1, stopping_thresh=1e-4,
                 plot_predictions=False, plot_wait_for_input=False):
        self.num_dists = num_dists
        self.num_iter = num_iter

        self.dists = [SkewGMM.SkewGaussian() for i in range(self.num_dists)]

        self.evaluator = evaluator
        self.update_rate = update_rate

        self.use_annealing = use_annealing
        self.annealing_v = annealing_start_v if use_annealing else 1
        self.annealing_step = (1 - self.annealing_v) / self.num_iter / 0.9

        self.log_lik = []
        self.stopping_thresh = stopping_thresh

        self.plot_predictions = plot_predictions
        self.plot_wait_for_input = plot_wait_for_input
        if self.plot_predictions:
            matplotlib.use("TkAgg")
            plt.ion()
            self.fig, self.ax = plt.subplots(2, 1)
            self.ax[0].set_title("Predictions plot")
            self.ax[1].set_title("Distributions plot")
            plt.draw()
            plt.show(block=False)

    def _initialize_dists(self, data):
        dim = data.shape[1]

        for dist_i in range(self.num_dists):
            dist = self.dists[dist_i]
            dist.ksi = data[np.random.choice(len(data))]
            dist.sigma = np.eye(dim) / 2
            dist.lambd = np.eye(dim) / np.sqrt(2)
            dist.w = 1 / self.num_dists

    def evaluate(self, data, iter_i):
        if self.evaluator is not None:
            prob_v, prob_dists = self.calculate_probs(data)
            pred = np.argmax(prob_dists, 1)
            log_lik = np.sum(np.log(prob_v)) if np.all(prob_v != 0) else -np.inf
            self.evaluator(iter_i, prob_dists, pred, log_lik)

        if self.plot_predictions:
            # Draw the predictions plot
            # self.ax[0].clear()
            # data_colors = np.clip(probs.T @ colors, 0, 1)
            # self.ax[0].scatter(data[:, 0], data[:, 1], color=data_colors, s=10)
            # self.ax[0].set_aspect('equal', 'box')
            # self.ax[0].set_title("Probabilities")

            # Draw the sample plot
            plt.sca(self.ax[1])
            self.ax[1].clear()
            values, _ = self.random_sample(1000)
            for dist_i in range(self.num_dists):
                sns.kdeplot(x=values[:, 0], y=values[:, 1], cmap="rocket",
                            shade=True, ax=self.ax[1],
                            bw_adjust=2, thresh=0, levels=20)
                plt.gca().set_aspect('equal')
            # self.ax[1].set_xlim(self.ax[0].get_xlim())
            # self.ax[1].set_ylim(self.ax[0].get_ylim())
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

        stopping_criterion_reached =\
            was_stopping_criterion_reached(self.log_lik, self.stopping_thresh)
        return stopping_criterion_reached

    def calculate_probs(self, data, annealing_v=1):
        log_probs = []
        for dist_i in range(self.num_dists):
            dist = self.dists[dist_i]
            dist_probs = dist.logpdf(data, include_w=True)
            log_probs.append(dist_probs)

        log_probs = np.array(log_probs)
        if annealing_v != 1:
            log_probs *= annealing_v

        # Rescale for numerical stability
        log_probs_max = np.max(log_probs, axis=0)
        log_probs -= log_probs_max
        probs = np.exp(log_probs)
        prob_v = np.sum(probs, axis=0)
        # Use the Bayes formula p(path|v) = p(v,path) / p(v)
        prob_dists = probs / (prob_v + SMALL_VALUE)
        prob_v *= np.exp(log_probs_max)

        return prob_v, prob_dists.T

    def fit(self, data):
        dim = data.shape[1]

        self._initialize_dists(data)

        # Update certian paramters
        for dist_i in range(self.num_dists):
            self.dists[dist_i]._update_params()

        self.evaluate(data, 0)

        for iter_i in range(self.num_iter):
            # Store the exp(z) in probs
            _, prob_dists = self.calculate_probs(
                data, annealing_v=self.annealing_v)

            w_sum = 0

            for dist_i in range(self.num_dists):
                dist = self.dists[dist_i]

                # Perform the E step
                # Initialize holders for the expectations
                exp_x = []
                exp_xx = []
                
                for data_i in range(len(data)):
                    # Calculate the expected parameters of the truncated normal
                    # distribution E(tau | y, z) = TN(mu, sigma, a)
                    # (equation 15)
                    mu = (dist.lambd.T @ dist.omega_inv @ (data[data_i] - dist.ksi))
                    sigma = dist.delta

                    def calc_Gr(r):
                        if dim <= 1:
                            return 1

                        index = np.arange(dim) != r
                        cov = sigma[index][:, index]
                        cov_part_r = sigma[index, r]
                        add_mean = np.linalg.inv(cov) @ cov_part_r * mu[r]
                        # mvn.mvnun is the cdf between given bounds
                        return mvn.mvnun(
                            lower=np.zeros(dim - 1),
                            upper=np.ones(dim - 1) * np.inf,
                            means=mu[index] + add_mean,
                            covar=cov
                        )[0]

                    def calc_Grs(r, s):
                        if dim <= 2:
                            return 1

                        index = np.arange(dim)
                        index = (index != r) & (index != s)
                        cov = sigma[index][:index]
                        cov_part_rs = sigma[index, [r, s]]
                        add_mean = np.linalg.inv(cov) @ cov_part_rs @ mu[[r, s]]
                        # mvn.mvnun is the cdf between given bounds
                        return mvn.mvnun(
                            lower=np.zeros(dim - 2),
                            upper=np.ones(dim - 2) * np.inf,
                            means=mu[index] + add_mean,
                            covar=cov
                        )[0]
    
                    # We have that a = 0. Therefore lower bound is zero
                    # mvn.mvnun is the cdf between given bounds
                    alpha = mvn.mvnun(
                        lower=np.zeros_like(mu),
                        upper=np.ones_like(mu) * np.inf,
                        means=mu, covar=sigma)[0] + SMALL_VALUE
    
                    # Calculate the E(tau)
                    #   (equation 10, 8)
                    fr = np.array([normal.pdf(0, mean=mu[r], cov=sigma[r, r], allow_singular=True)
                                  for r in range(dim)])
                    Gr = np.array([calc_Gr(r) for r in range(dim)])
                    q = fr * Gr
                    exp_x_i = (mu + 1 / alpha * sigma @ q).reshape([-1, 1])

                    # Calculate the E(tau tau.T)
                    #   (equation 11, 9)
                    frs = np.array([[
                        normal.pdf([0, 0], mean=mu[[r, s]], cov=sigma[[r, s]][:, [r, s]], allow_singular=True)
                        for r in range(len(mu))] for s in range(len(mu))])
                    Grs = np.array([[calc_Grs(r, s) for r in range(len(mu))]
                                    for s in range(len(mu))])

                    H = frs * Grs * (np.ones([dim, dim]) - np.eye(dim))
                    D = np.diag([
                        1 / sigma[r, r] * (-mu[r] * fr[r] * Gr[r] - (sigma @ H)[r, r])
                        for r in range(len(mu))])
                    mu = mu.reshape([-1, 1])
                    exp_xx_i = mu @ exp_x_i.T + exp_x_i @ mu.T - mu @ mu.T +\
                                  sigma + 1 / alpha * sigma @ (H + D) @ sigma

                    exp_x.append(exp_x_i.reshape([-1]))
                    exp_xx.append(exp_xx_i)

                exp_x = np.array(exp_x)
                exp_xx = np.array(exp_xx)

                # Perform the M step
                dist_probs = prob_dists[:, dist_i].reshape([-1, 1])

                w = np.mean(dist_probs)
                denom = dist_probs.sum()
                ksi = np.sum((data - (dist.lambd @ exp_x.T).T) * dist_probs, 0) / denom
                lambd_denom = np.linalg.pinv(np.sum(exp_xx * dist_probs.reshape([-1, 1, 1]), 0))
                lambd = ((data - ksi).T @ (exp_x * dist_probs)) @ lambd_denom
                normalized = (data - ksi - (lambd @ exp_x.T).T)
                sigma = (normalized.T @ (normalized * dist_probs) +
                         lambd @ ((exp_xx * dist_probs.reshape([-1, 1, 1])).sum(0) -
                                  exp_x.T @ (exp_x * dist_probs)) @ lambd) / denom

                rate = self.update_rate
                dist.w = dist.w * (1 - rate) + w * rate
                dist.ksi = dist.ksi * (1 - rate) + ksi * rate
                dist.lambd = dist.lambd * (1 - rate) + lambd * rate
                dist.sigma = dist.sigma * (1 - rate) + sigma * rate

                dist.sigma = make_spd(dist.sigma)

                w_sum += dist.w

            # Rescale the mixture probabilities
            for dist_i in range(self.num_dists):
                self.dists[dist_i].w /= w_sum

            # Perform a step of annealing
            if self.use_annealing:
                self.annealing_v = self.annealing_v + self.annealing_step
                self.annealing_v = float(np.clip(self.annealing_v, 0, 1))

            # Update certain parameter values
            for dist_i in range(self.num_dists):
                self.dists[dist_i]._update_params()

            stopping_criterion_reached = self.evaluate(data, iter_i + 1)
            if stopping_criterion_reached:
                break

        _, prob_dists = self.calculate_probs(data)
        return prob_dists

    def predict(self, data, output_probs=False):
        prob_v, prob_dists = self.calculate_probs(data)

        if output_probs:
            return prob_dists
        else:
            return np.argmax(prob_dists, 1)

    def random_sample(self, num):
        values = []
        dists = []

        for i in range(num):
            dist_i = np.random.choice(
                self.num_dists,
                p=[self.dists[i].w for i in range(self.num_dists)])
            dist = self.dists[dist_i]

            tau = np.abs(normal.rvs(cov=np.eye(len(dist.ksi))))
            value = normal.rvs(mean=dist.ksi, cov=dist.sigma) + dist.lambd @ tau

            values.append(value)
            dists.append(dist_i)

        return np.array(values), np.array(dists)

    class SkewGaussian:
        def __init__(self):
            self.ksi = None
            self.sigma = None
            self.lambd = None
            self.w = None

            # parameters updated during optimization
            self.omega = None
            self.omega_inv = None
            self.delta = None

        def _update_params(self):
            dim = len(self.ksi)
            self.omega = self.sigma + self.lambd @ self.lambd.T
            self.omega_inv = np.linalg.pinv(self.omega)
            self.delta = np.eye(dim) - self.lambd.T @ self.omega_inv @ self.lambd

        def logpdf(self, values, include_w=False):
            # Equation (1)
            dim = len(self.ksi)
            omega, omega_inv, delta = self.omega, self.omega_inv, self.delta

            probs = np.log(2) * dim
            probs += normal.logpdf(values, mean=self.ksi, cov=omega,
                                   allow_singular=True)
            probs += normal.logcdf(
                (self.lambd.T @ omega_inv @ (values - self.ksi).T).T,
                cov=delta, allow_singular=True)
            if include_w:
                probs += np.log(self.w)
            return probs.reshape([-1])

        def pdf(self, values, include_w=False):
            # Equation (1)
            dim = len(self.ksi)
            omega, omega_inv, delta = self.omega, self.omega_inv, self.delta

            probs = 2 ** dim * normal.pdf(values, mean=self.ksi, cov=omega)
            probs *= normal.cdf((self.lambd @ omega_inv @ (values - self.ksi).T).T,
                                 cov=delta)
            if include_w:
                probs *= self.w
            return probs.reshape([-1])
