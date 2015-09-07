from __future__ import division
from kern import Kern
import numpy as np
from sklearn.preprocessing import StandardScaler


class GPR(object):
    def __init__(self, X, y, kernel=None):
        self.X = X
        self.y = y

        self._noise_variance = 0.00001
        self._kernel = kernel
        self._scaler = StandardScaler(with_std=False)
        self._scaler.fit(self.y)
        self.y = self._scaler.transform(self.y)

        assert self._kernel is not None

    @property
    def noise_variance(self):
        return self._noise_variance

    @noise_variance.setter
    def noise_variance(self, value):
        self._noise_variance = value

    def predict(self, X_test):
        assert isinstance(self._kernel, Kern)

        K = self._kernel.K(self.X)
        K_star = self._kernel.K(self.X, X_test)
        K_star_star = self._kernel.K(X_test)

        L = np.linalg.cholesky(K + self._noise_variance * np.eye(len(K)))
        Lk = np.linalg.solve(L, K_star)
        mu = np.dot(Lk.T, np.linalg.solve(L, self.y))
        s2 = np.diag(K_star_star) - np.sum(Lk ** 2, axis=0) + self._noise_variance

        return mu + self._scaler.mean_, s2
