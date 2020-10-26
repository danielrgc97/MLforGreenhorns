#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

# 93bc2ff7-0d50-11eb-98c0-005056ad4f31.
# b4fbbfe2-0fa9-11eb-98c0-005056ad4f31.

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--epochs", default=50, type=int, help="Number of SGD iterations over the data")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Generate an artifical regression dataset
    data, target = sklearn.datasets.make_regression(n_samples=args.data_size, random_state=args.seed)

    # TODO: Append a constant feature with value 1 to the end of every input data
    X = np.ones([data.shape[0],data.shape[1]+1])
    X[:,:-1] = data

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data = sklearn.model_selection.train_test_split(X, test_size=args.test_size, random_state=args.seed)
    train_target, test_target = sklearn.model_selection.train_test_split(target, test_size=args.test_size, random_state=args.seed)
    X_train = train_data
    X_test = test_data
    t_train = train_target
    t_test = test_target

    # Generate initial linear regression weights
    weights = generator.uniform(size=train_data.shape[1])

    train_rmses, test_rmses = [], []
    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`.
        # For every `args.batch_size`, average their gradient, and update the weights.
        # A gradient for example (x_i, t_i) is `(x_i^T weights - t_i) * x_i`,
        # and the SGD update is `weights = weights - args.learning_rate * gradient`.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        
        grads = np.zeros([train_data.shape[1],args.batch_size])
        for i in range(train_data.shape[0]):
            gradient = np.dot(np.transpose(X_train[permutation[i]])@weights-t_train[permutation[i]],X_train[permutation[i]])
            iCorr = i % args.batch_size
            grads[:,iCorr] = gradient
            if iCorr == args.batch_size - 1:
                gradient = np.sum(grads,axis=1)/args.batch_size
                weights = weights - args.learning_rate * gradient

        # TODO: Append current RMSE on train/test to train_rmses/test_rmses.
        predictions_train = np.dot(X_train,weights)
        rmse_train = np.sqrt(sklearn.metrics.mean_squared_error(t_train,predictions_train))
        train_rmses.append(rmse_train)
        predictions_test = np.dot(X_test,weights)
        rmse_test = np.sqrt(sklearn.metrics.mean_squared_error(t_test,predictions_test))
        test_rmses.append(rmse_test)
        
        

    # TODO: Compute into `explicit_rmse` test data RMSE when
    # fitting `sklearn.linear_model.LinearRegression` on train_data.
    model = sklearn.linear_model.LinearRegression().fit(X_train[:,:-1],t_train)
    predictions = model.predict(X_test[:,:-1])
    explicit_rmse = np.sqrt(sklearn.metrics.mean_squared_error(t_test,predictions))
    sgd_rmse = rmse_test

    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(train_rmses, label="Train")
        plt.plot(test_rmses, label="Test")
        plt.xlabel("Iterations")
        plt.ylabel("RMSE")
        plt.legend()
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return test_rmses[-1], explicit_rmse

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    sgd_rmse, explicit_rmse = main(args)
    print("Test RMSE: SGD {:.2f}, explicit {:.2f}".format(sgd_rmse, explicit_rmse))