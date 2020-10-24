#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--dataset", default="boston", type=str, help="Standard sklearn dataset to load")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    dataset = getattr(sklearn.datasets, "load_{}".format(args.dataset))()

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    X_train, X_test = sklearn.model_selection.train_test_split(dataset.data, test_size=args.test_size, random_state=args.seed)
    

    # TODO: Process the input columns in the following way:
    #
    # - if a column has only integer values, consider it a categorical column
    #   (days in a week, dog breed, ...; in general integer values can also
    #   represent numerical non-categorical values, but we use this assumption
    #   for the sake of an exercise). Encode the values with one-hot encoding
    #   using `sklearn.preprocessing.OneHotEncoder` (note that its output is by
    #   default sparse, you can use `sparse=False` to generate dense output;
    #   also use `handle_unknown="ignore"` to ignore missing values in test set).
    #
    # - for the rest of the columns, normalize their values so that they
    #   have mean 0 and variance 1; use `sklearn.preprocessing.StandardScaler`.
    #
    # In the output, there should be first all the one-hot categorical features,
    # and then the real-valued features. To process different dataset columns
    # differently, you can use `sklearn.compose.ColumnTransformer`.
    X_train_ord = np.ones([X_train.shape[0],1])
    for i in range(X_train.shape[1]):
        if np.all(X_train[:,i]%1 == 0):
            X_train_ord = np.concatenate((X_train_ord, np.reshape(X_train[:,i],(-1,1))), axis=1)
    onehotCols = X_train_ord.shape[1] - 1
    for i in range(X_train.shape[1]):
        if np.any(X_train[:,i]%1 != 0):
            X_train_ord = np.concatenate((X_train_ord, np.reshape(X_train[:,i],(-1,1))), axis=1)
    X_train_ord = X_train_ord[:,1:]

    X_test_ord = np.ones([X_test.shape[0],1])
    for i in range(X_test.shape[1]):
        if np.all(X_test[:,i]%1 == 0):
            X_test_ord = np.concatenate((X_test_ord, np.reshape(X_test[:,i],(-1,1))), axis=1)
    onehotCols = X_test_ord.shape[1] - 1
    for i in range(X_test.shape[1]):
        if np.any(X_test[:,i]%1 != 0):
            X_test_ord = np.concatenate((X_test_ord, np.reshape(X_test[:,i],(-1,1))), axis=1)
    X_test_ord = X_test_ord[:,1:]

    if onehotCols > 0:
        print("onehotCols", onehotCols)
        ct = sklearn.compose.ColumnTransformer([("norm1", sklearn.preprocessing.OneHotEncoder(sparse=False,handle_unknown='ignore'), slice(onehotCols)),("norm2", sklearn.preprocessing.StandardScaler(), slice(onehotCols,X_train_ord.shape[1]))])
    else:
        print("oooo")
        ct = sklearn.compose.ColumnTransformer([("norm2", sklearn.preprocessing.StandardScaler(), slice(onehotCols,X_train_ord.shape[1]))])
    

    # TODO: Generate polynomial features of order 2 from the current features.
    # If the input values are [a, b, c, d], you should generate
    # [a^2, ab, ac, ad, b^2, bc, bd, c^2, cd, d^2]. You can generate such polynomial
    # features either manually, or using
    # `sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)`.
    poly = sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)

    # TODO: You can wrap all the feature processing steps into one transformer
    # by using `sklearn.pipeline.Pipeline`. Although not strictly needed, it is
    # usually comfortable.
    pipe = sklearn.pipeline.Pipeline([('columT', ct), ('polinomial', poly)])

    # TODO: Fit the feature processing steps on the training data.
    # Then transform the training data into `train_data` (you can do both these
    # steps using `fit_transform`), and transform testing data to `test_data`.
    # train_data = pipe.fit_transform(X_train_ord)
    # test_data = pipe.fit_transform(X_test_ord)
    train_data = X_train
    test_data = X_test
    return train_data, test_data


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_data, test_data = main(args)
    for dataset in [train_data, test_data]:
        for line in range(min(dataset.shape[0], 5)):
            a = 0
            print(" ".join("{:.4g}".format(dataset[line, column]) for column in range(min(dataset.shape[1], 60))))