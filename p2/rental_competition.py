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

class Dataset:
    """Rental Dataset.
    The dataset instances consist of the following 12 features:
    0- season (1: winter, 2: sprint, 3: summer, 4: autumn)
    1- year (0: 2011, 1: 2012)
    2- month (1-12)
    3- hour (0-23)
    4- holiday (binary indicator)
    5- day of week (0: Sun, 1: Mon, ..., 6: Sat)
    6- working day (binary indicator; a day is neither weekend nor holiday)
    7- weather (1: clear, 2: mist, 3: light rain, 4: heavy rain)
    8- temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1)
    9- feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1)
    10- relative humidity (0-1 range)
    11- windspeed (normalized to 0-1 range)
    The target variable is the number of rentals in the given hour.
    """
    def __init__(self,
                 name="rental_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name))
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)

##CUSTOM-TRANSFORMER##
# class ExperimentalTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
#   def __init__(self):
#     print("custom transformer called")

#   def fit(self, X, y = None):
#     return self

#   def transform(self, X, y = None):
#     X_ = X.copy() # creating a copy to avoid changes to original dataset
#     feature = np.ones([X_.shape[0],1])
#     feature[:,0] = (X_[:,3]*X_[:,5])**2
#     X_ = np.concatenate((X_, np.reshape(feature[:,0],(-1,1))), axis=1)
#     return X_
##CUSTOM-TRANSFORMER##

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")
parser.add_argument("--test_size", default=0.1, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")

def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        
        # TODO: Train a model on the given dataset and store it in `model`.
        ct = sklearn.compose.ColumnTransformer([("norm1", sklearn.preprocessing.OneHotEncoder(sparse=False,handle_unknown='ignore'), [0,1,2,3,4,5,6,7]),("norm2", sklearn.preprocessing.StandardScaler(), slice(8,13))])
        poly = sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)
        pipe = sklearn.pipeline.Pipeline([('columT', ct), ('polinomial', poly),('estimator',sklearn.linear_model.SGDRegressor(max_iter=1000))])
        
        ##TESTING##
        X_train, X_test = sklearn.model_selection.train_test_split(train.data, test_size=args.test_size, random_state=args.seed)
        t_train, t_test = sklearn.model_selection.train_test_split(train.target, test_size=args.test_size, random_state=args.seed)
        pipe.fit(X_train,t_train)
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(t_test,pipe.predict(X_test)))
        print(rmse)
        ##TESTING##

        ##PLOTING##
        # if args.plot:
        #     import matplotlib.pyplot as plt
        #     for i in range(7):
        #         sold = []
        #         for j in range(24):
        #             ventas = 0
        #             divisor = 0
        #             for k in range(train.data.shape[0]):
        #                 if train.data[k,3] == j and train.data[k,5] == i:
        #                     ventas += train.target[k]
        #                     divisor+=1
        #             if divisor != 0:
        #                 ventas = ventas/divisor
        #             sold.append(ventas)
        #         plt.plot(sold, label=i)
        #         plt.legend()
        #     if args.plot is True: plt.show()
        #     else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")
        ##PLOTING##

        pipe.fit(train.data,train.target)
        model = pipe

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

