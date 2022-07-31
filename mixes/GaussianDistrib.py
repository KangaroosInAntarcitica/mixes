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
