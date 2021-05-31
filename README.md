# Scalable and Interpretable Point Processes

Pytorch implementation of the Variational Inference-Decomposable Point process model (VI-DPP) from our paper **Scalable and Interpretable Point Processes**.

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
The user can define the following arguments before training the model
* --dataset: Daraset name; acceptable values [mimic, mooc, retweet, stackOverflow]. Type: string. Default=mimic
* --Q: Number of past events taking into account for computing the data likelihood. Type: integer. Default=5
* --dataset: Daraset name; acceptable values [mimic, mooc, retweet, stackOverflow]. Type: string. Default=mimic
* --dataset: Daraset name; acceptable values [mimic, mooc, retweet, stackOverflow]. Type: string. Default=mimic
* --dataset: Daraset name; acceptable values [mimic, mooc, retweet, stackOverflow]. Type: string. Default=mimic
* --dataset: Daraset name; acceptable values [mimic, mooc, retweet, stackOverflow]. Type: string. Default=mimic
* --dataset: Daraset name; acceptable values [mimic, mooc, retweet, stackOverflow]. Type: string. Default=mimic
* --dataset: Daraset name; acceptable values [mimic, mooc, retweet, stackOverflow]. Type: string. Default=mimic
* --dataset: Daraset name; acceptable values [mimic, mooc, retweet, stackOverflow]. Type: string. Default=mimic
* --dataset: Daraset name; acceptable values [mimic, mooc, retweet, stackOverflow]. Type: string. Default=mimic
* --dataset: Daraset name; acceptable values [mimic, mooc, retweet, stackOverflow]. Type: string. Default=mimic
