# Scalable and Interpretable Point Processes

Pytorch implementation of the Variational Inference-Decomposable Point Process model (VI-DPP).

## Install

The `vi_dpp` package must be be installed to run examples. To install it, run 

    python setup.py vi_dpp

Other dependencies must be installed using the `requirement.txt` file.

## Train and test VI-DPP model on real-world datasets

To train and test the performance of VI-DPP use the script `train.py` in the `example` folder as follows:

    python train.py --dataset=mimic --Q=1 --max_epochs=100

The script firstly trains the VI-DPP model over the `mimic` dataset using Q=1 (See definition of arguments below) and  setting the maximum number of epochs equal to 100.
After running the script, details about the given dataset will be printed out accompanied with the predictive performance of the model on a test dataset in terms of average log-likelihood, root mean squared error (RMSE), and F1 score.

## Arguments in `train.py`
The user can define the following arguments before training the model:
* --dataset: Daraset name; acceptable values [mimic, mooc, retweet, stackOverflow]. Type: string. Default=mimic
* --Q: Number of past events taking into account for computing the data likelihood. Type: integer. Default=5
* --num_samples: Number of Monte Carlo samples used for evaluating ELBO. Type: integer. Default=1
* --num_weights: Number of weights used for importance sampling of ELBO evaluation. Type: integer. Default=1
* --threshold: Threshold used to determine if two consecutive values of ELBO are close; its value is used for convergence check. Type: float. Default=1
* --batch_size: The number of sequences considered at each iteration, i.e. the batch size. Type: integer. Default=32
* --max_epochs: Maximum number of epochs used for training the model. Type: integer. Default=1000
* --print_every: Number of epochs needed to be elapsed so various metrics to be printed. Type: integer. Default=5
* --patience: If 'patience' epochs are elapsed without improvement of the ELBO then stop optimization. Type: integer. Default=100
* --use_prior: Whether the prior (and posterior) of betas and alpha (variational parameters) is taken into account for computing ELBO. Type: boolean. Default=False
* --seed: Set seed for random number generator. Type: integer. Default=0
* --verbose: Set verbose mode. Type: boolean. Default=True
