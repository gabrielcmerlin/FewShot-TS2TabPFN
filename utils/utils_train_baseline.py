"""Baseline classifier heads for the few-shot experiment (replace TabPFN in exp/few_baseline/main.py).

Both heads are trained on the pre-extracted features of the support set (the same features TabPFN receives in
exp/few/main.py), so only the classifier differs from the TabPFN run (same features, same support sets, same test sets).
The scaler is fitted on the support set only, so the baselines see exactly what TabPFN sees.

    ridge: StandardScaler(with_mean=False) -> RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
           (the head of aeon's MultiRocketClassifier, aeon/classification/convolution_based/_multirocket.py)
    1nn:   StandardScaler -> 1-nearest-neighbour with Euclidean distance

Neither has predict_proba, so the "probabilities" are one-hot predictions (like aeon does for ridge) and AUROC/log-loss
are not meaningful. A support set with a single class cannot train either head: the prediction is then that class for
every test series (constant predictor) and the caller logs it. Times are measured around `fit` and `predict` only
(no CodeCarbon).
"""
import time

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODELS = ("ridge", "1nn")


def build_classifier(model):
    if model == "ridge":
        return make_pipeline(StandardScaler(with_mean=False), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)))
    if model == "1nn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1, algorithm="brute"))
    raise ValueError(f"Modelo desconhecido: '{model}'. Opções válidas: {list(MODELS)}")


def _one_hot(y_pred, classes):
    """No predict_proba: one-hot predictions (columns = classes)."""
    proba = np.zeros((len(y_pred), len(classes)))
    proba[np.arange(len(y_pred)), np.searchsorted(classes, y_pred)] = 1.0
    return proba


def train_test_model_baseline(model, X_train, X_test, y_train):
    """Fit on the support set and predict the test set. Returns
    (y_pred, y_proba, train_time, predict_time, predict_proba_time, test_time, constant_fallback), times in seconds;
    constant_fallback is True when the support set has a single class (every test series gets that class)."""
    classes = np.unique(y_train)
    if len(classes) < 2:
        y_pred = np.full(len(X_test), classes[0])
        return y_pred, _one_hot(y_pred, classes), 0.0, 0.0, 0.0, 0.0, True

    classifier = build_classifier(model)

    start = time.perf_counter()
    classifier.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = classifier.predict(X_test)
    predict_time = time.perf_counter() - start

    start = time.perf_counter()
    y_proba = _one_hot(y_pred, classifier.classes_)
    predict_proba_time = time.perf_counter() - start

    return y_pred, y_proba, train_time, predict_time, predict_proba_time, predict_time + predict_proba_time, False
