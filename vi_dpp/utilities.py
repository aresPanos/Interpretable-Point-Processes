import numpy as np
import pickle
from sklearn.metrics import f1_score
from vi_dpp import VI_model
from pomegranate import *

def load_dataset(full_path: str, seed: int=None, train_size=0.7, val_size=0.1, test_size=0.2, shuffle: bool=True):
    """
    Dataset loading function
    Arguments:
    ----------
    full_path: (str) Full path name of the dataset location
    seed: (int) Seed value of the random number generator
    train_size: (float) Proportion of the original dataset which is used for training
    val_size: (float) Proportion of the original dataset which is used for validation
    test_size: (float) Proportion of the original dataset which is used for testing
    shuffle: (bool) Falog indicating whether to shuffle or not the original dataset
    """

    if train_size < 0 or val_size < 0 or test_size < 0:
        raise ValueError("train_size, val_size, and test_size must be >= 0.")
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size, val_size, and test_size must add up to 1.")

    with open(full_path, 'rb') as f:
        full_data = pickle.load(f)

    times_all, marks_all = full_data['timestamps'], full_data['types']

    N = len(times_all)
    if seed is not None:
        np.random.seed(seed)
    all_idx = np.arange(N)
    np.random.shuffle(all_idx)

    train_end = int(train_size * N)
    val_end = int((train_size + val_size) * N)

    train_idx = all_idx[:train_end]
    val_idx = all_idx[train_end:val_end]
    test_idx = all_idx[val_end:]

    times_list_train, marks_list_train = [], []
    times_list_val, marks_list_val = [], []
    times_list_test, marks_list_test = [], []

    for idx in train_idx:
        times_list_train.append(times_all[idx])
        marks_list_train.append(marks_all[idx])

    for idx in val_idx:
        times_list_val.append(times_all[idx])
        marks_list_val.append(marks_all[idx])

    for idx in test_idx:
        times_list_test.append(times_all[idx])
        marks_list_test.append(marks_all[idx])

    return times_list_train, marks_list_train, times_list_val, marks_list_val, times_list_test, marks_list_test, max([mark.max() for mark in marks_all]) + 1


def log_likelihood_marks(times_list: list, marks_list: list, alpha, beta_mtr, delta_vec, Gamma_mtr, Q: int=200):
    """
    Computation of the log likelihood of marks
    Arguments:
    ----------
    times_list: (list) A list of ndarrays containing the event occurence times
    marks_list: (list) A list of ndarrays containing the corresponding event marks
    alpha: (float) The excitation factor
    delta_vec: The U x 1 background propability vector
    beta_mtr: (ndarrays)  A U x U matrix that contains the decay rates
    Gamma_mtr: (ndarrays) A U x U stochastic matrix that contains the conversion rates
    Q: (int) Proportion of the original dataset which is used for testing
    """
    log_lkl_org = 0.
    jumps_N = 0
    for s in range(len(times_list)):
        times = times_list[s]
        marks = marks_list[s]
        N = times.size
        jumps_N += N
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

    return log_lkl_org


def pos_constraint(input):
    """
    Softplus function
    Arguments:
    ----------
    input: scalar or ndarray
    """
    return np.log(1. + np.exp(input))


def compute_f1_score(times_list: list, marks_list: list, alpha, beta_mtr, delta_vec, Gamma_mtr, Q: int=200):
    """
    Computation of F1 score
    Arguments:
    ----------
    times_list: (list) A list of ndarrays containing the event occurence times
    marks_list: (list) A list of ndarrays containing the corresponding event marks
    alpha: (float) The excitation factor
    delta_vec: The U x 1 background propability vector
    beta_mtr: (ndarrays)  A U x U matrix that contains the decay rates
    Gamma_mtr: (ndarrays) A U x U stochastic matrix that contains the conversion rates
    Q: (int) Proportion of the original dataset which is used for testing
    """
    S = len(times_list)
    U_dim = delta_vec.size
    true_marks = np.zeros(S, dtype=np.int32)
    predicted_marks = np.zeros(S, dtype=np.int32)

    for s in range(S):
        times = times_list[s]
        marks = marks_list[s]
        true_marks[s] = marks[-1]
        N = times.size
        ar_x = np.arange(N)
        prob_all = np.zeros((N, U_dim))
        prob_all[0] = delta_vec
        Q_ = min(Q, N-1)
        Q_mtr = np.zeros((Q_, N-1))

        for u in range(U_dim):
            for q in range(Q_):
                marks_pred = u * np.ones(N -1 -q, dtype=np.int32)
                tij = times[:N-q-1] - times[q+1:N]
                Q_mtr[q, q:] = Gamma_mtr[marks[:-1-q], marks_pred] * np.exp(beta_mtr[marks[:-1-q], marks_pred] * tij)

            prob_all[1:, u] = delta_vec[u] + alpha * Q_mtr.sum(0)
        prob_all /= prob_all.sum(axis=1, keepdims=True)

        predicted_marks[s] = np.argmax(prob_all, axis=1)[-1]

    return f1_score(true_marks, predicted_marks, average='macro')


def compute_model_mode(model_vi: VI_model):
    """
    Computation of the mode for each variational distribution
    Arguments:
    ----------
    model_vi: VI_model
    """

    x_opt = model_vi.coeffs.detach().numpy()

    m_betas_alpha_opt = x_opt[:(model_vi.U_dim**2 + 1)]
    sigma_betas_alpha_opt = pos_constraint(x_opt[(model_vi.U_dim**2 + 1):2*(model_vi.U_dim**2 + 1)])
    concent_deltas_Gammas_opt = pos_constraint(x_opt[2*(model_vi.U_dim**2 + 1):]).reshape(model_vi.U_dim + 1, model_vi.U_dim)

    mode_alpha = np.exp(m_betas_alpha_opt[-1] - sigma_betas_alpha_opt[-1]**2)
    mode_betas = np.exp(m_betas_alpha_opt[:-1] - sigma_betas_alpha_opt[:-1]**2).reshape(model_vi.U_dim, model_vi.U_dim)
    mean_deltas_Gammas = concent_deltas_Gammas_opt / concent_deltas_Gammas_opt.sum(1)[:, None]

    return mode_alpha, mode_betas, mean_deltas_Gammas


def compute_scores_marks(model_vi: VI_model, times_list: list, marks_list: list):
    """
    Computation of the log-likelihood of marks and F1 score over the dataset (times_list, marks_list)
    Arguments:
    ----------
    model_vi: VI_model
    times_list: (list) A list of ndarrays containing the event occurence times
    marks_list: (list) A list of ndarrays containing the corresponding event marks
    """
    mode_alpha, mode_betas, mean_deltas_Gammas = compute_model_mode(model_vi)

    log_lkl_marks = log_likelihood_marks(times_list, marks_list, mode_alpha, mode_betas, mean_deltas_Gammas[0], mean_deltas_Gammas[1:], Q=model_vi.Q)
    f1_score = compute_f1_score(times_list, marks_list, mode_alpha, mode_betas, mean_deltas_Gammas[0], mean_deltas_Gammas[1:], Q=model_vi.Q)

    return log_lkl_marks, f1_score


def mle_lognormal(times: np.ndarray):
    """
    Maximum likelihood estimation of the parameters of a mixture of log-Normal distributions
    Arguments:
    ----------
    times: (ndarray) A ndarray containing all the event occurence times for all sequences
    """
    if times.size < 2:
        lognorm = LogNormalDistribution(times.mean(), 1)
        mode_lognorm = np.exp(times.mean() - 1)
    else:
        lognorm = LogNormalDistribution(times.mean(), times.var())
        lognorm.fit(times)
        mode_lognorm = np.exp(lognorm.parameters[0] - lognorm.parameters[1]**2)

    return mode_lognorm, lognorm


def mode_lognormal(times_list: list, marks_list: list, U: int):
    """
    Computation of the mode of a log-Normal distribution (for each mark) after
    finding the optimized parameters by MLE over the data (times_list, marks_list)
    Arguments:
    ----------
    times_list: (list) A list of ndarrays containing the event occurence times
    marks_list: (list) A list of ndarrays containing the corresponding event marks
    U: (int) Dimension of the mark space
    """

    inter_arrival = np.concatenate([(times[1:] - times[:-1]) for times in times_list])
    inter_arrival[inter_arrival <= 0] += 1e-8 # For numerical stability
    marks = np.concatenate([marks[:-1] for marks in marks_list])

    modes = np.ones(U)
    models_vec = np.zeros(U, dtype=object)
    for u in range(U):
        data = inter_arrival[marks == u]
        if data.size > 0:
                modes[u], models_vec[u] = mle_lognormal(data)

    return modes, models_vec


def compute_scores_times(times_list: list, marks_list: list, times_list_val: list, marks_list_val: list, U: int):
    """
    Computation of the times log-likelihood and RMSE over the dataset (times_list_val, marks_list_val).
    We use (times_list, marks_list) for training U log-normal distributions, one for each mark
    Arguments:
    ----------
    times_list: (list) A list of ndarrays containing the event occurence times used for training
    marks_list: (list) A list of ndarrays containing the corresponding event marks used for training
    times_list_val: (list) A list of ndarrays containing the event occurence times used for computing the two scores
    marks_list_val: (list) A list of ndarrays containing the corresponding event marks used for computing the two scores
    U: (int) Dimension of the mark space
    """

    modes, models_vec = mode_lognormal(times_list, marks_list, U)

    log_lkl_times = 0.
    predicted_taus = np.zeros(len(times_list_val))
    true_taus = np.zeros(len(times_list_val))
    for s in range(len(times_list_val)):
        times = times_list_val[s]
        marks = marks_list_val[s]
        inter_arrival = times[1:] - times[:-1]
        inter_arrival[inter_arrival <= 0.] = 1e-8
        marks_inter = marks[:-1]

        true_taus[s] = inter_arrival[-1]
        predicted_taus[s] = modes[marks[-2]]

        for u in range(U):
            data = inter_arrival[marks_inter == u]
            if data.size > 0 and models_vec[u] != 0.:
                log_lkl_times += models_vec[u].log_probability(data[:, None]).sum()

    rmse = np.sqrt(np.square((predicted_taus - true_taus) / true_taus).mean())

    return log_lkl_times, rmse


def compute_all_scores(model_vi: VI_model, times_list: list, marks_list: list, times_list_val: list, marks_list_val: list, U: int):
    """
    Computation of the mean log-likelihood (times and marks included), F1 score, and RMSE over the dataset (times_list_val, marks_list_val).
    We use (times_list, marks_list) for training
    Arguments:
    ----------
    model_vi: VI_model
    times_list: (list) A list of ndarrays containing the event occurence times used for training
    marks_list: (list) A list of ndarrays containing the corresponding event marks used for training
    times_list_val: (list) A list of ndarrays containing the event occurence times used for computing the two scores
    marks_list_val: (list) A list of ndarrays containing the corresponding event marks used for computing the two scores
    U: (int) Dimension of the mark space
    """

    log_lkl_times, rmse = compute_scores_times(times_list, marks_list, times_list_val, marks_list_val, model_vi.U_dim)
    log_lkl_marks, f1_score = compute_scores_marks(model_vi, times_list_val, marks_list_val)

    num_events = sum(times_seq.size for times_seq in times_list_val)

    return (log_lkl_times + log_lkl_marks) / num_events, rmse, f1_score
