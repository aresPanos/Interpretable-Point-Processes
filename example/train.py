import numpy as np
import argparse
import os
from vi_dpp import VI_model, load_dataset, compute_all_scores

def run_experiment(args_in):
    print('============\nConfiguration\n============')
    print(args_in)
    print('============\n')

    dir_dataset = os.path.abspath(__file__ + "/../../data/") + '/' + args_in.dataset + '.pkl'
    times_list_tr, marks_list_tr, times_list_val, marks_list_val, times_list_tst, marks_list_tst, U_dim = load_dataset(dir_dataset, seed=args_in.seed)
    print(7*'=')
    print('Dataset: ', args_in.dataset)
    print('Number of events Train: {}  Val: {}  Test: {} ' .format(sum(time_seq.size for time_seq in times_list_tr), sum(time_seq.size for time_seq in times_list_val), sum(time_seq.size for time_seq in times_list_tst)))
    print('U: ', U_dim)
    print(7*'=')

    model = VI_model(times_list_tr, marks_list_tr, times_list_val, marks_list_val,
                     U_dim=U_dim, Q=args_in.Q, num_samples=args_in.num_samples, num_weights=args_in.num_weights,
                     max_epochs=args_in.max_epochs, batch_size=args_in.batch_size, interval=args_in.interval,
                     print_every=args_in.print_every, use_prior=args_in.use_prior, verbose=args_in.verbose)

    print('\nTraining VI-DPP...')
    model.fit(seed=args_in.seed)
    print('Done!')

    print('\nTraining log-Normal model...')
    log_lkl, rmse, f1_score = compute_all_scores(model, times_list_tr, marks_list_tr, times_list_tst, marks_list_tst, model.U_dim)
    print('Done!')

    print('Training time: %.4f seconds' %model.train_time)

    print('\nPredictive performance')
    print('=======================')
    print('average log-lkl: {:.4f} | F1: {:.4f} | RMSE: {:.4f}' .format(log_lkl, f1_score, rmse))
    print('Done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mimic', help="Daraset name; acceptable values [mimic, mooc, retweet, stackOverflow].")
    parser.add_argument('--Q', type=int, default=5, help="Number of past events taking into account for computing the data likelihood.")
    parser.add_argument('--num_samples', type=int, default=1, help="Number of Monte Carlo samples used for evaluating ELBO.")
    parser.add_argument('--num_weights', type=int, default=1, help="Number of weights used for importance sampling of ELBO evaluation.")
    parser.add_argument('--threshold', type=float, default=1., help="Threshold used to determine if two consecutive values of ELBO are close; its value is used for convergence check.")
    parser.add_argument('--batch_size', type=int, default=32, help="the number of sequences considered at each iteration, i.e. the batch size.")
    parser.add_argument('--max_epochs', type=int, default=2000, help="Maximum number of epochs used for training the model.")
    parser.add_argument('--interval', type=int, default=100, help="Number of epochs needed to be elapsed so the prior hyperparameters to be updated.")
    parser.add_argument('--print_every', type=int, default=5, help="Number of epochs needed to be elapsed so various metrics to be printed.")
    parser.add_argument('--patience', type=int, default=100, help="If 'patience' epochs are elapsed without improvement of the ELBO then stop optimization.")
    parser.add_argument('--use_prior', type=bool, default=False, help="Whether the prior (and posterior) of betas and alpha (variational parameters) is taken into account for computing ELBO.")
    parser.add_argument('--seed', type=int, default=0, help="Set seed for random number generator.")
    parser.add_argument('--verbose', type=bool, default=True, help="Set verbose mode.")

    args = parser.parse_args()

    run_experiment(args)
