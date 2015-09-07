import os
import numpy as np
from rdkit import Chem
from rpy2.robjects.packages import importr
from numpy.lib.stride_tricks import as_strided

rchemcpp = importr("Rchemcpp")


class Kern(object):
    def __init__(self):
        pass

    def K(self, X, X2=None):
        raise NotImplemented


class RBF(Kern):
    def __init__(self, length_scale, variance, labels=None):
        super(RBF, self).__init__()
        self.length_scale = length_scale
        self.variance = variance
        self.labels = labels

    def K(self, X, X2=None):
        """
        # Copyright (c) 2012, GPy authors (see GPy_AUTHORS.txt).
        # Licensed under the BSD 3-clause license (see GPy_LICENSE.txt)
        """
        r = self._unscaled_dist(X, X2) / self.length_scale

        return self.variance * np.exp(-0.5 * r ** 2)

    def _unscaled_dist(self, X, X2=None):
        """
        # Copyright (c) 2012, GPy authors (see GPy_AUTHORS.txt).
        # Licensed under the BSD 3-clause license (see GPy_LICENSE.txt)
        """
        X = X[self.labels]

        if X2 is not None:
            X2 = X2[self.labels]

        # X, = self._slice_X(X)
        if X2 is None:
            Xsq = np.sum(np.square(X), 1)
            r2 = -2. * np.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])

            # force diagnoal to be zero: sometime numerically a little negative
            as_strided(r2, shape=(r2.shape[0],), strides=((r2.shape[0] + 1) * r2.itemsize,))[:, ] = 0.

            r2 = np.clip(r2, 0, np.inf)
            return np.sqrt(r2)
        else:
            # X2, = self._slice_X(X2)
            X1sq = np.sum(np.square(X), 1)
            X2sq = np.sum(np.square(X2), 1)
            r2 = -2. * np.dot(X, X2.T) + X1sq[:, None] + X2sq[None, :]
            r2 = np.clip(r2, 0, np.inf)
            return np.sqrt(r2)


class CombinationKern(Kern):
    def __init__(self, kern_a, kern_b, alpha=0.5, func=np.add):
        assert isinstance(kern_a, Kern)
        assert isinstance(kern_b, Kern)
        self.kern_a = kern_a
        self.kern_b = kern_b
        self.alpha = alpha
        self.func = func

    def K(self, X, X2=None):
        if self.alpha == 1:
            return self.kern_a.K(X, X2)
        elif self.alpha == 0:
            return self.kern_b.K(X, X2)

        K_a = self.kern_a.K(X, X2)
        K_b = self.kern_b.K(X, X2)

        return self.func(np.multiply(self.alpha, K_a), np.multiply(1 - self.alpha, K_b))


class RChemKern(Kern):
    def __init__(self, X, X_test, label='mol', temp_dir='./', alpha=1, silentMode=True):
        super(RChemKern, self).__init__()
        self.X = X
        self.X_test = X_test
        self.label = label
        self.train_sdf_path = os.path.abspath(temp_dir + '_train.sdf')
        self.test_sdf_path = os.path.abspath(temp_dir + '_test.sdf')
        self._sdf_created = False
        self._params = {
            'returnNormalized': False,
            'silentMode': silentMode
        }
        self.alpha = alpha
        self._rchem_func = rchemcpp.sd2gramSpectrum

    def K(self, X, X2=None):
        if not self._sdf_created:
            self._create_sdf()

        if X2 is None:
            if X is self.X:
                return self.alpha * np.array(self._rchem_func(self.train_sdf_path, **self._params))

            if X is self.X_test:
                return self.alpha * np.array(self._rchem_func(self.test_sdf_path, **self._params))
        else:
            if X is self.X and X2 is self.X_test:
                return self.alpha * np.array(self._rchem_func(self.train_sdf_path, self.test_sdf_path, **self._params))

        raise ValueError

    def _create_sdf(self):
        if self._sdf_created:
            return

        for df, file_path in zip([self.X, self.X_test], [self.train_sdf_path, self.test_sdf_path]):
            w = Chem.SDWriter(file_path)
            try:
                for mol in df[self.label]:
                    w.write(mol)
                w.flush()
            finally:
                w.close()

        self._sdf_created = True


class MinMaxTanimoto(RChemKern):
    def __init__(self, X, X_test, label='mol', temp_dir='./', alpha=1, silentMode=True):
        super(MinMaxTanimoto, self).__init__(X, X_test, label, temp_dir, alpha, silentMode)
        self._params['kernelType'] = 'minmaxTanimoto'
        self._rchem_func = rchemcpp.sd2gramSpectrum


class Tanimoto(RChemKern):
    def __init__(self, X, X_test, label='mol', temp_dir='./', alpha=1, silentMode=True):
        super(Tanimoto, self).__init__(X, X_test, label, temp_dir, alpha, silentMode)
        self._params['kernelType'] = 'tanimoto'
        self._rchem_func = rchemcpp.sd2gramSpectrum


class Pharmacore(RChemKern):
    def __init__(self, X, X_test, label='mol', temp_dir='./', alpha=1, silentMode=True):
        super(Pharmacore, self).__init__(X, X_test, label, temp_dir, alpha, silentMode)
        self._rchem_func = rchemcpp.sd2gram3Dpharma


class CombinationKern(Kern):
    def __init__(self, kern_a, kern_b, alpha=0.5, func=np.add):
        assert isinstance(kern_a, Kern)
        assert isinstance(kern_b, Kern)
        self.kern_a = kern_a
        self.kern_b = kern_b
        self.alpha = alpha
        self.func = func

    def K(self, X, X2=None):
        if self.alpha == 1:
            return self.kern_a.K(X, X2)
        elif self.alpha == 0:
            return self.kern_b.K(X, X2)

        K_a = self.kern_a.K(X, X2)
        K_b = self.kern_b.K(X, X2)

        return self.func(np.multiply(self.alpha, K_a), np.multiply(1 - self.alpha, K_b))
