from __future__ import division
from __future__ import print_function
import itertools
import numpy as np
from datetime import datetime
import GPy
from scipy.stats import norm
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import cross_validation

features = [
    u'JCHEM_ACIDIC_PKA',
    u'JCHEM_LOGP',
    u'JCHEM_POLARIZABILITY',
    u'JCHEM_POLAR_SURFACE_AREA',
    u'JCHEM_REFRACTIVITY',
]


def predict(X_train, y_train, X_test, lengthscale, variance):
    k = GPy.kern.RBF(len(features), variance=variance, lengthscale=lengthscale)
    model = GPy.models.GPRegression(X_train.as_matrix(), y_train.as_matrix(), kernel=k, normalizer=None)

    mu, var = model.predict(X_test.as_matrix())

    return mu, np.sqrt(var)


def compute_fold(X, y, train_fold, test_fold, arr_lengthscale, arr_variance):
    results = []

    for (p_ls, p_var) in itertools.product(arr_lengthscale, arr_variance):
        mu_, s_ = predict(X.iloc[train_fold], y.iloc[train_fold], X.iloc[test_fold], p_ls, p_var)

        r2 = r2_score(y.iloc[test_fold], mu_)
        mse = mean_squared_error(y.iloc[test_fold], mu_)
        mll = np.sum(
            [norm.logpdf(t[0], loc=t[1], scale=t[2]) for t in zip(y.iloc[test_fold].values, mu_.flatten(), s_)])

        results.append({'ls': p_ls, 'var': p_var, 'r2': r2, 'mse': mse, 'mll': mll})

    return results


def cross_validation_scoring(X, y, arr_lengthscale, arr_variance, n_folds=3):
    k_fold = cross_validation.KFold(len(X), n_folds=n_folds)

    results = []

    for i, (train_fold, test_fold) in enumerate(k_fold):
        print('Training on Fold', '{}/{}'.format(i + 1, n_folds))
        started = datetime.now()
        fold_result = compute_fold(X, y, train_fold, test_fold, arr_lengthscale, arr_variance)
        ended = datetime.now()
        print('Training on Fold', '{}/{}'.format(i + 1, n_folds), 'Completed -', ended - started)

        results.extend(fold_result)

    return DataFrame(results)


def calculate_scores(df, n_folds):
    grid_len = len(df) // n_folds
    grid_scores = []

    for offset in range(0, grid_len):
        ls = df.iloc[offset]['ls']
        var = df.iloc[offset]['var']

        avg_r2 = 0
        avg_mse = 0
        avg_mll = 0

        for fold_start in range(0, len(df), grid_len):
            avg_r2 += df.iloc[fold_start + offset]['r2']
            avg_mse += df.iloc[fold_start + offset]['mse']
            avg_mll += df.iloc[fold_start + offset]['mll']

        avg_r2 /= n_folds
        avg_mse /= n_folds
        avg_mll /= n_folds

        grid_scores.append((ls, var, avg_r2, avg_mse, avg_mll))

    df = DataFrame(grid_scores, columns=['ls', 'var', 'avg_r2', 'avg_mse', 'avg_mll'])

    return df


def plot_results(y_test, mu, s, title='New Model', allowed_error=None, labels=None):
    ax = plt.gca()
    plt.xlabel('Measured (log)')
    plt.ylabel('Predicted (log)')
    plt.scatter(y_test, mu)
    plt.axis('equal')

    ylim = plt.ylim()
    plt.errorbar(y_test, mu, yerr=s, fmt='o', elinewidth=1, capthick=1, capsize=3, lolims=False, uplims=False)
    plt.ylim(*ylim)

    if labels is not None:
        for x, y, label in zip(y_test, mu, labels):
            ann = ax.annotate(label, (x, y))

    xlim = plt.xlim()
    plt.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], 'r--', linewidth=.3)
    plt.xlim(*xlim)

    min_ = round(min(y_test))
    max_ = max(y_test) + 1
    plt.xticks(np.arange(round(min(y_test)), max(y_test) + 1, 1.0))
    plt.yticks(np.arange(round(min(y_test)), max(y_test) + 1, 1.0))

    plt.xlim([min_, max_])

    if allowed_error:
        ideal = np.linspace(xlim[0], xlim[1])
        plt.gca().fill_between(ideal, ideal - allowed_error, ideal + allowed_error, color='#DDDDDD', alpha=0.5)

    plt.grid(True)
    plt.title(title)
