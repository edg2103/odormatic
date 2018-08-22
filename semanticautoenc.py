import numpy as np
import scipy.sparse as sp
from scipy import linalg
from scipy import sparse
import sklearn.externals.six as six 
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from abc import ABCMeta, abstractmethod
from sklearn.utils import check_array, check_X_y, deprecated, as_float_array
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.sparsefuncs import mean_variance_axis, inplace_column_scale
from sklearn.utils.fixes import sparse_lsqr
from sklearn.utils.seq_dataset import ArrayDataset, CSRDataset
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing.data import normalize as f_normalize
import numbers

"""
implements "Semantic Autoencoder for Zero-Shot Learning" by Kodirov, Xiang, & Gong 2017, CVPR
"""
def _preprocess_data(X, y, fit_intercept, normalize=False, copy=True,
                     sample_weight=None, return_mean=False):
    """
    Centers data to have mean zero along axis 0. If fit_intercept=False or if
    the X is a sparse matrix, no centering is done, but normalization can still
    be applied. The function returns the statistics necessary to reconstruct
    the input data, which are X_offset, y_offset, X_scale, such that the output
        X = (X - X_offset) / X_scale
    X_scale is the L2 norm of X - X_offset. If sample_weight is not None,
    then the weighted mean of X and y is zero, and not the mean itself. If
    return_mean=True, the mean, eventually weighted, is returned, independently
    of whether X was centered (option used for optimization with sparse data in
    coordinate_descend).
    This is here because nearly all linear models will want their data to be
    centered. This function also systematically makes y consistent with X.dtype
    """

    if isinstance(sample_weight, numbers.Number):
        sample_weight = None

    X = check_array(X, copy=copy, accept_sparse=['csr', 'csc'],
                    dtype=FLOAT_DTYPES)
    y = np.asarray(y, dtype=X.dtype)

    if fit_intercept:
        if sp.issparse(X):
            X_offset, X_var = mean_variance_axis(X, axis=0)
            if not return_mean:
                X_offset[:] = X.dtype.type(0)

            if normalize:

                # TODO: f_normalize could be used here as well but the function
                # inplace_csr_row_normalize_l2 must be changed such that it
                # can return also the norms computed internally

                # transform variance to norm in-place
                X_var *= X.shape[0]
                X_scale = np.sqrt(X_var, X_var)
                del X_var
                X_scale[X_scale == 0] = 1
                inplace_column_scale(X, 1. / X_scale)
            else:
                X_scale = np.ones(X.shape[1], dtype=X.dtype)

        else:
            X_offset = np.average(X, axis=0, weights=sample_weight)
            X -= X_offset
            if normalize:
                X, X_scale = f_normalize(X, axis=0, copy=False,
                                         return_norm=True)
            else:
                X_scale = np.ones(X.shape[1], dtype=X.dtype)
        y_offset = np.average(y, axis=0, weights=sample_weight)
        y = y - y_offset
    else:
        X_offset = np.zeros(X.shape[1], dtype=X.dtype)
        X_scale = np.ones(X.shape[1], dtype=X.dtype)
        if y.ndim == 1:
            y_offset = X.dtype.type(0)
        else:
            y_offset = np.zeros(y.shape[1], dtype=X.dtype)

    return X, y, X_offset, y_offset, X_scale


class AutoEnc(six.with_metaclass(ABCMeta, BaseEstimator)):
    @abstractmethod
    def fit(self, X, y):
        """Fit model."""

    def _decision_function(self, X,inv=False):
#        check_is_fitted(self, "coef_")

        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
      #  return safe_sparse_dot(X, self.coef_.T,
        if not inv:
            return safe_sparse_dot(self.coef_, X.T,
                               dense_output=True).T + self.intercept_
        else:
          if len(self.intercept_.shape)==1:
            int0 = self.intercept_.reshape(self.intercept_.shape[0],1)
          else:
            int0 = self.intercept_
          return safe_sparse_dot(self.coef_.T, X.T - int0,
                               dense_output=True) #+ self.intercept_

    _preprocess_data = staticmethod(_preprocess_data)
    
    def predict(self, X, inv=False):
        """Predict using the linear model
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.
        Returns
        -------
        C : array, shape = (n_samples,)
            Returns predicted values.
        """
        return self._decision_function(X,inv=inv)


    def _set_intercept(self, X_offset, y_offset, X_scale):
        """Set the intercept_
        """
        if self.fit_intercept:
            self.coef_ = self.coef_ / X_scale
            self.intercept_ = y_offset - np.dot(X_offset, self.coef_.T)
        else:
            self.intercept_ = np.zeros(1)

class SemanticAutoEnc(AutoEnc, RegressorMixin):
    def __init__(self, fit_intercept=True, normalize=False, copy_X=True,
                 n_jobs=1,lambd=1):
        self.lambd = lambd
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.intercept_ = np.zeros(1)

    _preprocess_data = staticmethod(_preprocess_data)
    
    def fit(self, X, y, sample_weight=None):   
        lambd = self.lambd 
        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
            copy=self.copy_X, sample_weight=sample_weight)
        A = np.dot(y.T,y);
        B = lambd*np.dot(X.T,X);
        C = (1+lambd)*np.dot(y.T,X);
        W = linalg.solve_sylvester(A,B,C);
        self.coef_ = W
        self._set_intercept(X_offset, y_offset, X_scale)
        return self