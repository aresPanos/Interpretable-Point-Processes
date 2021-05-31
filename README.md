# Scalable and Interpretable Point Processes

Pytorch implementation of the Variational Inference-Decomposable Point process model (VI-DPP) from our paper **Scalable and Interpretable Point Processes**.

## Install

The `vi_dpp` package must be be installed to run examples. To install it, run 

    python setup.py  vi_dpp

Other dependencies must be installed using the `requirement.txt` file.

## Train and test VI-DPP model on real-world datasets

An example script is provided in the `examples` folder. To run the example, run the script `script_run_example.py` as follows:

    python script_run_example.py -d . -p params.json -o output.json

The script will read the experiment parameters in `params.json` that holds the parameters to simulate a realization of approx. 2000 events from a multivariate Hawkes process with 50 dimensions, and run all the learning algorithms discussed in the paper. Relevant performance evaluation information is printed along the run of the script.
