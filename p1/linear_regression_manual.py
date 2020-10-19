#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

# 93bc2ff7-0d50-11eb-98c0-005056ad4f31.
# b4fbbfe2-0fa9-11eb-98c0-005056ad4f31.

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.1, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    
    # Load Boston housing dataset
    dataset = sklearn.datasets.load_boston()

    # TODO: Append a new feature to all input data, with value "1"
    X = np.ones([dataset.data.shape[0],dataset.data.shape[1]+1])
    X[:,:-1] = dataset.data

    # TODO: Split the dataset into a train set and a test set.
    X_train, X_test = sklearn.model_selection.train_test_split(X, test_size=args.test_size, random_state=args.seed)
    t_train, t_test = sklearn.model_selection.train_test_split(dataset.target, test_size=args.test_size, random_state=args.seed)

    # TODO: Solve the linear regression using the algorithm from the lecture,
    XtX = np.dot(np.transpose(X_train),X_train)
    XtXInv = np.linalg.inv(XtX)
    XtXInvXt = np.dot(XtXInv,np.transpose(X_train))
    w = np.dot(XtXInvXt,t_train)
    
    # TODO: Predict target values on the test set
    predictions = np.dot(X_test,w)

    # TODO: Compute root mean square error on the test set predictions
    error = predictions - t_test
    N = error.shape[0]
    rmse = np.sqrt(sklearn.metrics.mean_squared_error(t_test,predictions))

    return rmse

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    rmse = main(args)
    print("{:.2f}".format(rmse))