# Gaussian Heterogeneus Open-set Theory (GHOST)
This repository contains an implementation of GHOST.
GHOST builds gaussian models of each feature dimension from samples of each class and leverages their Z-scores at inference time to normalize classification logits.

## Setup
### While in development
Just add this repo (containing GHOST.py) to your PYTHONPATH

## Use
See the driver script for usage examples.
Otherwise, create a GHOST model by calling:

`GHOST_model =  GHOST(filtered_train_logits, filtered_train_FV)`

Where arguments are the corresponding logits and penultimate features from a pre-trained backbone (for only correctly predicted samples).

At inference time, call:

`probs = GHOST_model.ReScore(test_logits, test_FVs)`

GHOST will return a 1-D tensor with a probability of knowness for each sample (corresponding to the zeroth dimension of input logits/features)
