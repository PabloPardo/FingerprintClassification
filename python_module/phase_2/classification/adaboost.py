import inspect
from numpy.core.umath import sign
import numpy as np
import warnings
from sklearn.externals import six
from joblib import Parallel, delayed
from time import time


class DecisionStump():
    def __init__(self, n_steps=40):
        """
        :param n_steps: Number of steps used to grid search
                        over the threshold value of the decision
                        stump.
        """
        self.n_steps = n_steps
        self.weights = []
        self.best_stump = {}
        self.min_error = np.inf
        self.best_class_est = []

    def predict(self, X, dim, thresh_val, thresh_ineq):
        """
        Weak binary classifier which classifies the instances using
        one of the features and a threshold.

        :param X: Matrix with the instances.
        :param dim: Feature Dimensions to use in the classification.
        :param thresh_val: Threshold value to classify the instances.
        :param thresh_ineq: Inequality to use in the decision stamp,
                            the default value is 'bt' (bigger than),
                            but can be set to 'lt'.

        :return: Array filled with 1 and -1
        """
        ret_array = np.ones((np.shape(X)[0], 1))
        if thresh_ineq == 'lt':
            ret_array[X[:, dim] <= thresh_val] = -1.0
        else:
            ret_array[X[:, dim] > thresh_val] = -1.0
        return ret_array

    def fit(self, X, y, weights):
        """
        Finds a stump that best classifies the data, iterating
        over the threshold value, the feature dimension and the
        inequality direction.

        :param X: Array with the input data.
        :param y: Real labels of the input data.
        :param weights: Weight array used to search over the
                       best stamps.

        :return: best stump, minimal error and best prediction.
        """
        data_matrix = np.mat(X)
        label_mat = np.mat(y).T
        m, n = np.shape(data_matrix)
        self.best_class_est = np.mat(np.zeros((m, 1)))
        self.weights = weights
        ones = np.mat(np.ones((m, 1)))

        for i in range(n):
            range_min = data_matrix[:,i].min()
            range_max = data_matrix[:, i].max()
            step_size = (range_max-range_min)/self.n_steps

            for j in range(-1, int(self.n_steps)+1):
                thresh_val = (range_min + float(j) * step_size)

                for inequal in ['lt', 'gt']:
                    predicted_vals = self.predict(data_matrix, i, thresh_val, inequal)

                    # err_arr = np.mat(np.ones((m, 1)))
                    err_arr = ones.copy()
                    err_arr[predicted_vals == label_mat] = 0
                    # err_arr[(predicted_vals == 1) & (label_mat == 1)] = -0.5
                    # err_arr[(predicted_vals == -1) & (label_mat == 1)] = 1
                    err_arr[(predicted_vals == 1) & (label_mat == -1)] = 3

                    weighted_error = np.dot(weights.T, err_arr)
                    # print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, thresh_val, inequal, weighted_error)
                    if weighted_error < self.min_error:
                        min_error = weighted_error
                        self.best_class_est = predicted_vals.copy()
                        self.best_stump['dim'] = i
                        self.best_stump['thresh'] = thresh_val
                        self.best_stump['ineq'] = inequal

        return self.best_stump, self.min_error, self.best_class_est

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep: boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and w[0].category == DeprecationWarning:
                    # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

            # XXX: should we rather test if instance of estimator?
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        args, varargs, kw, default = inspect.getargspec(init)
        if varargs is not None:
            raise RuntimeError("scikit-learn estimators should always "
                               "specify their parameters in the signature"
                               " of their __init__ (no varargs)."
                               " %s doesn't follow this convention."
                               % (cls, ))
        # Remove 'self'
        # XXX: This is going to fail if the init is a staticmethod, but
        # who would do this?
        args.pop(0)
        args.sort()
        return args

    def set_params(self, **params):
        """Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The former have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        Returns
        -------
        self
        """
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        for key, value in six.iteritems(params):
            split = key.split('__', 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if not name in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s' %
                                     (name, self))
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if not key in valid_params:
                    raise ValueError('Invalid parameter %s ' 'for estimator %s'
                                     % (key, self.__class__.__name__))
                setattr(self, key, value)
        return self


class Adaboost():
    def __init__(self, estimator=None, params=None, n_iter=40, n_jobs=1, verbose=0):
        if not estimator:
            self.estimator = DecisionStump()
        else:
            self.estimator = estimator

        if not params:
            params = {'n_steps': 40}

        self.estimator.set_params(**params)
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.weak_clfs = []
        self.confidence = []
        self.alpha = 0
        self.weights = []

    def fit(self, X, y):
        m = np.shape(X)[0]
        self.weights = np.mat(0.01*np.ones((m, 1))/m)
        self.confidence = np.mat(np.zeros((m, 1)))

        for i in range(self.n_iter):
            best_stump, error, class_est = self.estimator.fit(X, y, self.weights)
            # print "weigh_arr:", weigh_arr.T

            self.alpha = float(0.5*np.ma.log((1.0-error)/max(error, 1e-16)))
            best_stump['alpha'] = self.alpha
            self.weak_clfs.append(best_stump)
            # print "class_est: ", class_est.T

            expon = np.ma.multiply(-1*self.alpha*np.mat(y).T, class_est)
            self.weights = np.ma.multiply(self.weights, np.ma.exp(expon))
            self.weights = self.weights/self.weights.sum()
            self.confidence += self.alpha*class_est
            # print "agg_class_est: ", agg_class_est.T

            agg_errors = np.ma.multiply(sign(self.confidence) != np.mat(y).T, np.ones((m, 1)))
            error_rate = agg_errors.sum()/m
            # print "Iteration {0} ---- total error: {1}".format(i, error_rate)

            if error_rate == 0.0:
                break

    def predict(self, X, ineq='lt', thresh=0, prob=False):
        data_matrix = np.mat(X)
        m = np.shape(data_matrix)[0]
        agg_class_est = np.mat(np.zeros((m, 1)))
        bin_class_est = np.mat(np.zeros((m, 1)))
        for i in range(len(self.weak_clfs)):
            class_est = self.estimator.predict(data_matrix, self.weak_clfs[i]['dim'],
                                               self.weak_clfs[i]['thresh'],
                                               self.weak_clfs[i]['ineq'])
            agg_class_est += self.weak_clfs[i]['alpha']*class_est
            # print agg_class_est

        if ineq == 'lt':
            bin_class_est[agg_class_est <= thresh] = -1
            bin_class_est[agg_class_est > thresh] = 1
        else:
            bin_class_est[agg_class_est > thresh] = -1
            bin_class_est[agg_class_est <= thresh] = 1

        if prob:
            return bin_class_est, agg_class_est
        else:
            return bin_class_est

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep: boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and w[0].category == DeprecationWarning:
                    # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

            # XXX: should we rather test if instance of estimator?
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        args, varargs, kw, default = inspect.getargspec(init)
        if varargs is not None:
            raise RuntimeError("scikit-learn estimators should always "
                               "specify their parameters in the signature"
                               " of their __init__ (no varargs)."
                               " %s doesn't follow this convention."
                               % (cls, ))
        # Remove 'self'
        # XXX: This is going to fail if the init is a staticmethod, but
        # who would do this?
        args.pop(0)
        args.sort()
        return args

    def set_params(self, **params):
        """Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The former have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        Returns
        -------
        self
        """
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        for key, value in six.iteritems(params):
            split = key.split('__', 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if not name in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s' %
                                     (name, self))
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if not key in valid_params:
                    raise ValueError('Invalid parameter %s ' 'for estimator %s'
                                     % (key, self.__class__.__name__))
                setattr(self, key, value)
        return self

    def score(self, X, y, sample_weight=None):
        """Returns the mean accuracy on the given test data and labels.
        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.
        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.
        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        from .metrics import accuracy_score
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)