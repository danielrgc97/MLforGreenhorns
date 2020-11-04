#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import urllib.request

import numpy as np
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.linear_model

# 93bc2ff7-0d50-11eb-98c0-005056ad4f31.
# b4fbbfe2-0fa9-11eb-98c0-005056ad4f31.
# 2eff3afe-1393-11eb-8e81-005056ad4f31.

class Dataset:
    """Thyroid Dataset.
    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features
    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """
    def __init__(self,
                 name="thyroid_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name))
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="thyroid_competition.model", type=str, help="Model path")
parser.add_argument("--test_size", default=0.1, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")

def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        # np.savetxt('../../../Desktop/x.txt',train.data)
        # np.savetxt('../../../Desktop/t.txt',train.target)

        # TODO: Train a model on the given dataset and store it in `model`.
        ct = sklearn.compose.ColumnTransformer([("norm1", sklearn.preprocessing.OneHotEncoder(sparse=False,handle_unknown='ignore'), slice(15)),("norm2", sklearn.preprocessing.StandardScaler(), slice(15,21))])
        pipe = sklearn.pipeline.Pipeline([
            ('ct', ct), 
            ('poly', sklearn.preprocessing.PolynomialFeatures(include_bias=False)),
            ('lg',sklearn.linear_model.LogisticRegression(random_state=args.seed))])
        param_grid = {
            'poly__degree': [2],
            'lg__C': [126],
        }
        search = sklearn.model_selection.GridSearchCV(pipe, param_grid, n_jobs=-1, cv=5)

        ##TESTING##
        # X_train, X_test = sklearn.model_selection.train_test_split(train.data, test_size=args.test_size, random_state=args.seed)
        # t_train, t_test = sklearn.model_selection.train_test_split(train.target, test_size=args.test_size, random_state=args.seed)
        # search.fit(X_train, t_train)
        # print("Best parameter (CV score=%0.3f):" % search.best_score_)
        # print(search.best_params_)
        # predictions = search.predict(X_test)
        # aciertos = 0
        # for i in range(predictions.shape[0]):
        #     if predictions[i] == t_test[i]: aciertos += 1
        # test_accuracy = aciertos/predictions.shape[0]
        # print("TEST ACCURACY:", test_accuracy)
        ##TESTING##

        search.fit(train.data, train.target)
        model = search

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, as either a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)