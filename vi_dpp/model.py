import numpy as np
import torch
import time
from sklearn.metrics import f1_score
torch.set_default_tensor_type(torch.DoubleTensor)


class VI_model(object):
    """
    Importance weighted variational objective function
    Arguments:
    ----------
    data_times: (list) A list of ndarrays containing the event occurence times used for training
    data_marks: (list) A list of ndarrays containing the event marks used for training
    data_times_val: (list) A list of ndarrays containing the event occurence times used for validation
    data_marks_val: (list) A list of ndarrays containing the event marks used for validation
    U_dim: (int) Dimension of the mark space
    Q: (int) Number of past events taking into account for computing the data likelihood. i.e. the value of Q
    num_samples: (int) Number of Monte Carlo samples used for evaluating ELBO
    num_weights: (int) Number of weights used for importance sampling of ELBO evaluation
    threshold: (float) Threshold used to determine if two consecutive values of ELBO are close; its value is used for convergence check
    weight_temp: (float) Value used for tempering the updated values
    lr_gamma: (float) hyperparameter of the Exponential learning rate scheduler
    batch_size: (int) the number of sequences considered at each iteration, i.e. the batch size
    max_epochs: (int) maximum number of epochs
    interval: (int) Number of epochs needed to be elapsed so the prior hyperparameters to be updated
    print_every: (int) Number of epochs needed to be elapsed so various metrics to be printed
    patience: (int) If 'patience' epochs are elapsed without improvement of the ELBO then stop optimization
    use_prior: (bool) Whether the prior (and posterior) of betas and alpha (variational parameters) is taken into account for computing ELBO
    """

    def __init__(self,
                 data_times: list,
                 data_marks: list,
                 data_times_val: list,
                 data_marks_val: list,
                 U_dim: int,
                 Q: int,
                 num_samples: int,
                 num_weights: int,
                 threshold=1.,
                 weight_temp=1.,
                 lr_gamma=0.9999,
                 batch_size: int=64,
                 max_epochs: int=100,
                 interval: int=5,
                 print_every: int=1,
                 patience: int=100,
                 use_prior: bool=False,
                 verbose: bool=False):

        if not isinstance(data_times[0], np.ndarray):
            raise TypeError("Invalid 'data_times' provided")

        if not isinstance(data_marks[0], np.ndarray):
            raise TypeError("Invalid 'data_marks' provided")

        self.num_samples = num_samples
        self.num_weights = num_weights
        self.threshold = threshold
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.interval = interval
        self.print_every = print_every
        self.weight_temp = weight_temp
        self.lr_gamma = lr_gamma
        self.patience = patience
        self.use_prior = use_prior
        self.verbose = verbose
        self.Q = Q

        self.U_dim = U_dim
        self.n_params = 1 + (2*self.U_dim + 1) * self.U_dim
        self.n_var_params = 2 + (3*self.U_dim + 1) * self.U_dim
        self.size_alpha_betas = 1 + self.U_dim**2
        self.data_marks = data_marks
        self.num_seqs = len(data_marks)
        self.num_jumps = [mark_vec.size for mark_vec in data_marks]
        self.elbo_history = np.zeros(self.max_epochs)
        self.log_sqrt2pi = 0.5 * np.log(2 * np.pi)
        self.data_times_val = data_times_val
        self.data_marks_val = data_marks_val

        self.cache_data(data_times, Q)

        self.positive_constraint = torch.nn.Softplus()
        self.coeffs = None
        self.coeffs_old = None
        self.prior_sigma_betas_alpha = torch.ones(self.size_alpha_betas)


    def cache_data(self, data_times, Q_in):
        """
        Cache computations needed for optimization
        """
        self.tau_ij_minus_list = []
        self.Q_list = []
        for i in range(self.num_seqs):
            vec_times = data_times[i]
            N = vec_times.size
            Q = min(Q_in, N - 1)
            minus_tau_ij = np.zeros((Q, N - 1))
            for q in range(Q):
                minus_tau_ij[q, q:N-1] = vec_times[:N-q-1] - vec_times[q+1:N] # change sign - used in likelihood computation

            self.tau_ij_minus_list.append(torch.tensor(minus_tau_ij))
            self.Q_list.append(Q)


    def check_convergence(self):
        """
        Check if the algorithm converged
        """
        if self.best_loss -  self.loss_val < self.threshold:
            self.impatient += 1
        else:
            self.best_loss = self.loss_val
            self.impatient = 0

        if self.impatient == self.patience:
           return True
        return False


    def callback(self, it, coeffs=None, end=""):
        """
        Print values of interest (ELBO, log-lkl, F1, training time)'
        """
        if (it+1) % self.print_every == 0 or it==0:
            if coeffs is None:
                print('Epoch: {:d}/{:d} | ELBO: {:.4f} | dx: {:.4f}' .format(it+1, self.max_epochs, self.elbo_history[it], self.max_norm))
            else:
                log_lkl_val, f1_val = self.compute_metrics_val(coeffs)
                print('Epoch: {:d}/{:d} | ELBO: {:.4f}  | log-lkl-val: {:.4f}  | F1 val: {:.4f} | training time: {:.4f}s'
                     .format(it+1, self.max_epochs, self.elbo_history[it], log_lkl_val, f1_val, self.train_time))


    def compute_metrics_val(self, x_opt):
        """
        Compute the value of log-likelihood nad the F1 score over the validation dataset
        """
        m_betas_alpha_opt = x_opt[:(self.U_dim**2 + 1)]
        sigma_betas_alpha_opt = np.log(1. + np.exp(x_opt[(self.U_dim**2 + 1):2*(self.U_dim**2 + 1)]))
        concent_deltas_Gammas_opt = np.log(1. + np.exp(x_opt[2*(self.U_dim**2 + 1):])).reshape(self.U_dim + 1, self.U_dim)

        mode_alpha = np.exp(m_betas_alpha_opt[-1] - sigma_betas_alpha_opt[-1]**2)
        mode_betas = np.exp(m_betas_alpha_opt[:-1] - sigma_betas_alpha_opt[:-1]**2).reshape(self.U_dim, self.U_dim)
        mean_deltas_Gammas = concent_deltas_Gammas_opt / concent_deltas_Gammas_opt.sum(1)[:, None]

        log_lkl_tst, f1_tst = self.log_likelihood_acc(mode_alpha, mode_betas, mean_deltas_Gammas[0], mean_deltas_Gammas[1:])

        return log_lkl_tst, f1_tst


    def log_likelihood_acc(self, alpha, beta_mtr, delta_vec, Gamma_mtr):
        """
        Helper function to compute the value of log-likelihood nad the F1 score over the validation dataset
        """
        log_lkl_org = 0.
        Q = max(self.Q_list)
        sum_N = 0
        predicted_marks = np.zeros(len(self.data_times_val), dtype=np.int32)
        true_marks = np.zeros(len(self.data_times_val), dtype=np.int32)
        for s in range(len(self.data_times_val)):
            times = self.data_times_val[s]
            marks = self.data_marks_val[s]
            true_marks[s] = marks[-1]
            N = times.size
            ar_x = np.arange(N)
            Q_ = min(Q, N-1)
            Q_mtr = np.zeros((Q_, N-1))
            Q_denom = np.zeros((N-1, Q_))
            for q in range(Q_):
                tij = times[:N-q-1] - times[q+1:N]
                Q_mtr[q, q:] = Gamma_mtr[marks[:-1-q], marks[1+q:]] * np.exp(beta_mtr[marks[:-1-q], marks[1+q:]] * tij)
                Q_denom[q:, q] = (Gamma_mtr[marks[:-1-q]] * np.exp(beta_mtr[marks[:-1-q]] * tij[:, None])).sum(1)

            numer = np.log(delta_vec[marks[0]]) + np.log(delta_vec[marks[1:]] + alpha * Q_mtr.sum(0)).sum()
            denom = np.log(1 + alpha * Q_denom.sum(1)).sum()

            log_lkl_org += numer - denom

            prob_all = np.zeros((N, self.U_dim))
            prob_all[0] = delta_vec
            for u in range(self.U_dim):
                for q in range(Q_):
                    marks_pred = u * np.ones(N -1 -q, dtype=np.int32)
                    tij = times[:N-q-1] - times[q+1:N]
                    Q_mtr[q, q:] = Gamma_mtr[marks[:-1-q], marks_pred] * np.exp(beta_mtr[marks[:-1-q], marks_pred] * tij)

                prob_all[1:, u] = delta_vec[u] + alpha * Q_mtr.sum(0)
            prob_all /= prob_all.sum(axis=1, keepdims=True)

            predicted_marks[s] = np.argmax(prob_all, axis=1)[-1]

        return log_lkl_org, f1_score(true_marks, predicted_marks, average='macro')


    def stable_softmax(self, input):
        """
        Compute softmax values for a 1-d torch.Tensor 'input'.
        """
        e_x = (input - input.max()).exp()
        return e_x / e_x.sum()


    def log_posterior(self, betas_alpha, m_betas_alpha, sigma_betas_alpha):
        """
        Compute the value of log-posterior
        """
        betas_alpha_log = betas_alpha.log()
        log_pdf_betas_alpha = - 0.5 * (betas_alpha_log - m_betas_alpha).square() / sigma_betas_alpha.square() - (betas_alpha_log + sigma_betas_alpha.log() + self.log_sqrt2pi)

        return log_pdf_betas_alpha.sum()


    def log_likelihood(self, batch_inds, betas_alpha, deltas, Gammas):
        """
        Compute the value of log-likelihood
        """
        alpha =  betas_alpha[-1]
        beta_mtr = betas_alpha[:-1].reshape(self.U_dim, self.U_dim)
        log_lkl = 0.
        for s in batch_inds:
            marks = self.data_marks[s]
            minus_tau_ij = self.tau_ij_minus_list[s]
            N = self.num_jumps[s]
            Q = self.Q_list[s]

            Q_mtr = torch.zeros((Q, N-1))
            Q_denom = torch.zeros((N-1, Q))
            for q in range(Q):
                tij = minus_tau_ij[q, q:]
                marks_minus = marks[:-1-q]
                marks_plus = marks[1+q:]
                Q_mtr[q, q:] = Gammas[marks_minus, marks_plus] * (beta_mtr[marks_minus, marks_plus] * tij).exp()
                Q_denom[q:, q] = (Gammas[marks_minus] * (beta_mtr[marks_minus] * tij[:, None]).exp()).sum(1)

            numer = deltas[marks[0]].log() + (deltas[marks[1:]] + alpha * Q_mtr.sum(0)).log().sum()
            denom = (1. + alpha * Q_denom.sum(1)).log().sum()

            log_lkl += numer - denom

        return self.num_seqs * log_lkl / batch_inds.size


    def log_prior(self, betas_alpha):
        """
        Compute the value of log-prior
        """
        # log-pdf betas, alpha ~ N(0, C**2)
        log_pdf_betas_alpha = - 0.5 * betas_alpha.square() / self.prior_sigma_betas_alpha
        return log_pdf_betas_alpha.sum()


    def log_importance_weight(self, batch_inds, samples_betas_alpha_l_i, samples_deltas_l_i, samples_Gammas_l_i, m_betas_alpha, sigma_betas_alpha):
        """
        Compute the value of a single importance weight 'log(w_i)'
        """
        # Compute the log-posterior and log-prior
        logpost, logprior = 0. , 0.
        if self.use_prior:
            logpost = self.log_posterior(samples_betas_alpha_l_i, m_betas_alpha, sigma_betas_alpha)
            logprior = self.log_prior(samples_betas_alpha_l_i)

        # Compute the log-likelihood
        loglik = self.log_likelihood(batch_inds, samples_betas_alpha_l_i, samples_deltas_l_i, samples_Gammas_l_i)

        return loglik + logprior - logpost


    def objective_l(self, batch_inds, samples_betas_alpha_l, samples_deltas_l, samples_Gammas_l, m_betas_alpha, sigma_betas_alpha):
        log_w_arr = torch.zeros(self.num_weights)
        for i in range(self.num_weights):
            # Compute the importance weights (and their gradients)
            log_w_arr[i] = self.log_importance_weight(batch_inds, samples_betas_alpha_l[i], samples_deltas_l[i], samples_Gammas_l[i], m_betas_alpha, sigma_betas_alpha)
        # Temper the weights
        log_w_arr /= self.weight_temp
        w_tilde = self.stable_softmax(log_w_arr).detach()  # Detach 'w_tilde' from backward computations
        # Compute the weighted average over all `num_weights` samples
        value_i = w_tilde * log_w_arr
        return value_i.sum()


    def objective(self, x, batch_inds):
        """
        Importance weighted variational objective function
        Arguments:
        ----------
        x: (torch.Tensor) The variational parameters to be optimized
        batch_inds: (np.ndarray) The indexes of the selected batch
        """
        # Split the parameters into concentration parameters and means/stds
        m_betas_alpha = x[:self.size_alpha_betas]
        sigma_betas_alpha = self.positive_constraint(x[self.size_alpha_betas:2*self.size_alpha_betas])
        concent_deltas_Gammas = self.positive_constraint(x[2*self.size_alpha_betas:]).reshape(self.U_dim + 1, self.U_dim)

        # Sample noise and Dirichlet distributed probability vectors
        samples_betas_alpha = (m_betas_alpha + sigma_betas_alpha * torch.randn(self.num_samples, self.num_weights, self.size_alpha_betas)).exp()

        dirichlet_dist = torch.distributions.dirichlet.Dirichlet(concent_deltas_Gammas)
        samples_dirichlet = dirichlet_dist.rsample((self.num_samples, self.num_weights)) # num_samples x num_weights x U+1 x U

        samples_deltas = samples_dirichlet[:, :, 0, :] # num_samples x num_weights x U
        samples_Gammas = samples_dirichlet[:, :, 1:, :] # num_samples x num_weights x U x U

        elbo_value = 0.0
        # Compute a Monte Carlo estimate of ELBO
        for l in range(self.num_samples):
            elbo_value += self.objective_l(batch_inds, samples_betas_alpha[l], samples_deltas[l], samples_Gammas[l], m_betas_alpha, sigma_betas_alpha)
        elbo_value /= self.num_samples
        return elbo_value


    def hyperparameter_optimize(self, x, batch_inds):
        """
        Update prior's hyperparameters
        Arguments:
        ----------
        x: (torch.Tensor) The variational parameters to be optimized
        batch_inds: (np.ndarray) The indexes of the selected batch
        """
        log_w_arr = torch.zeros(self.num_weights)

        m_betas_alpha = x[:self.size_alpha_betas]
        sigma_betas_alpha = self.positive_constraint(x[self.size_alpha_betas:2*self.size_alpha_betas])
        concent_deltas_Gammas = self.positive_constraint(x[2*self.size_alpha_betas:]).reshape(self.U_dim + 1, self.U_dim)

        # Sample noise and Dirichlet distributed probability vectors
        samples_betas_alpha = (m_betas_alpha + sigma_betas_alpha * torch.randn(self.num_weights, self.size_alpha_betas)).exp()

        dirichlet_dist = torch.distributions.dirichlet.Dirichlet(concent_deltas_Gammas)
        samples_dirichlet = dirichlet_dist.sample((self.num_weights, )) # num_weights x U+1 x U

        samples_deltas = samples_dirichlet[:, 0, :] # num_weights x U
        samples_Gammas = samples_dirichlet[:, 1:, :] # num_weights x U x U

        opt_sigma_now = samples_betas_alpha.square()

        for i in range(self.num_weights):
            log_w_arr[i] = self.log_importance_weight(batch_inds, samples_betas_alpha[i], samples_deltas[i], samples_Gammas[i], m_betas_alpha, sigma_betas_alpha)

        log_w_arr /= self.weight_temp
        w_tilde = self.stable_softmax(log_w_arr).detach()
        opt_sigma = (w_tilde.unsqueeze(1) * opt_sigma_now).sum(0)

        self.prior_sigma_betas_alpha = 0.5 * (opt_sigma +  self.prior_sigma_betas_alpha)


    def fit(self, coeffs_0=None, seed=None):
        """
        Training the model
        Arguments:
        ----------
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        if coeffs_0 is None:
            coeffs_0 = np.zeros(2 + (self.U_dim + 1) * self.U_dim + 2 * self.U_dim**2)
            coeffs_0[:(self.U_dim**2 + 1)] = np.random.normal(loc=2.1, scale=0.1, size=self.U_dim**2 + 1)
            coeffs_0[(self.U_dim**2 + 1):2*(self.U_dim**2 + 1)] = np.log(np.exp(np.clip(np.random.normal(loc=0.2, scale=0.1, size=self.U_dim**2 + 1), 1e-1, 2.0)) - 1.)
            coeffs_0[2*(self.U_dim**2 + 1):] = np.log(np.exp(0.5 + 2*np.random.rand((self.U_dim + 1) * self.U_dim)) - 1.)

        if not isinstance(coeffs_0, np.ndarray):
            raise TypeError("Invalid 'coeffs_0' provided. Coeffs must be a numpy array")

        if coeffs_0.size != self.n_var_params or coeffs_0.ndim != 1:
            raise ValueError("Invalid size of 'coeffs_0' provided. 'coeffs_0' must be a numpy array of size (" + str(self.n_var_params) + ", )")


        self.coeffs = torch.tensor(coeffs_0, requires_grad=True)
        self.coeffs_old = self.coeffs.detach().clone()

        optimizer = torch.optim.Adam([self.coeffs], lr=0.03)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_gamma)
        self.train_time = 0.
        all_seqs = np.arange(self.num_seqs)
        self.impatient = 0
        self.best_loss = np.inf

        for epoch in range(self.max_epochs):
            start_t = time.time()
            # Variational parameters update
            rand_inds = np.random.permutation(self.num_seqs)
            for first in range(0, self.num_seqs, self.batch_size):
                final_ind = min(first + self.batch_size, self.num_seqs)
                batch_ind = rand_inds[first:final_ind]
                optimizer.zero_grad()
                elbo = -1.0 * self.objective(self.coeffs, batch_ind)
                elbo.backward()
                optimizer.step()
                scheduler.step()
            self.train_time += time.time() - start_t
            coeffs = self.coeffs.detach()
            coeffs_np = coeffs.numpy()
            self.elbo_history[epoch] = -elbo.detach().numpy()
            lkl_val, _ = self.compute_metrics_val(coeffs_np)
            self.loss_val = -1.0 * lkl_val

            # Check that no NaN values are present after the variational parameters update
            if torch.isnan(coeffs).any():
                raise ValueError('NaNs in coeffs! Stop optimization...')

            # Convergence check
            if self.check_convergence():
                if self.verbose:
                    self.callback(epoch, coeffs_np)
                    print('Converged!')
                break
            elif self.verbose:
                self.callback(epoch, coeffs_np)

            # Update hyper-parameters
            if (epoch+1) % self.interval == 0:
                start_t = time.time()
                self.hyperparameter_optimize(coeffs, all_seqs)
                self.train_time += time.time() - start_t
        print()
