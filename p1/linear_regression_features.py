#!/usr/bin/env python3
import argparse
import numpy as np
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

# 93bc2ff7-0d50-11eb-98c0-005056ad4f31.
# b4fbbfe2-0fa9-11eb-98c0-005056ad4f31.

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_size", default=40, type=int, help="Data size")
parser.add_argument("--range", default=3, type=int, help="Feature order range")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Create the data
    xs = np.linspace(0, 7, num=args.data_size)
    ys = np.sin(xs) + np.random.RandomState(args.seed).normal(0, 0.2, size=args.data_size)

    rmses = []
    for order in range(1, args.range + 1):

        # TODO: Create features of x^1, ..., x^order.
        X = np.zeros([args.data_size,order])
        for i in range(order):
            X[:,i] = xs**(i+1)

        # TODO: Split the data into a train set and a test set.
        X_train, X_test = sklearn.model_selection.train_test_split(X, test_size=args.test_size, random_state=args.seed)
        t_train, t_test = sklearn.model_selection.train_test_split(ys, test_size=args.test_size, random_state=args.seed)

        # TODO: Fit a linear regression model using `sklearn.linear_model.LinearRegression`.
        model = sklearn.linear_model.LinearRegression()
        model.fit(X_train,t_train)

        # TODO: Predict targets on the test set using the trained model.
        predictions = model.predict(X_test)

        # TODO: Compute root mean square error on the test set predictions
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(t_test,predictions))

        rmses.append(rmse)

    return rmses

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    rmses = main(args)
    for order, rmse in enumerate(rmses):
        print("Maximum feature order {}: {:.2f} RMSE".format(order + 1, rmse))