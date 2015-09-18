from numpy.core.umath import sign
import numpy as np
from joblib import Parallel, delayed
from time import time


def stump_classify(data_matrix, dim, thresh_val, thresh_ineq):
    """
    Weak binary classifier which classifies the instances using
    one of the features and a threshold.

    :param data_matrix: Matrix with the instances.
    :param dim: Dimension to use in the classification.
    :param thresh_val: Threshold value to classify the instances.
    :param thresh_ineq: Inequality to use in the decision stamp,
                        the default value is 'bt' (bigger than),
                        but can be set to 'lt'.

    :return: Array filled with 1 and -1
    """
    ret_array = np.ones((np.shape(data_matrix)[0], 1))
    if thresh_ineq == 'lt':
        ret_array[data_matrix[:, dim] <= thresh_val] = -1.0
    else:
        ret_array[data_matrix[:, dim] > thresh_val] = -1.0
    return ret_array


def build_stump(data_arr, class_labels, weigh_arr, num_steps=10.0):
    """
    Finds a stump that best classifies the data, iterating
    over the threshold value, the feature dimension and the
    inequality direction.

    :param data_arr: Array with the input data.
    :param class_labels: Real labels of the input data.
    :param weigh_arr: Weight array used to search over the
                      best stamps.
    :param num_steps: Number of steps used to grid search
                      over the threshold value of the decision
                      stump.

    :return: best stump, minimal error and best prediction.
    """
    data_matrix = np.mat(data_arr)
    label_mat = np.mat(class_labels).T
    m, n = np.shape(data_matrix)
    best_stump = {}
    best_class_est = np.mat(np.zeros((m, 1)))
    min_error = np.inf
    ones = np.mat(np.ones((m, 1)))

    for i in range(n):
        range_min = data_matrix[:,i].min()
        range_max = data_matrix[:, i].max()
        step_size = (range_max-range_min)/num_steps

        for j in range(-1, int(num_steps)+1):
            thresh_val = (range_min + float(j) * step_size)

            for inequal in ['lt', 'gt']:
                predicted_vals = stump_classify(data_matrix, i, thresh_val, inequal)

                # err_arr = np.mat(np.ones((m, 1)))
                err_arr = ones.copy()
                err_arr[predicted_vals == label_mat] = 0
                # err_arr[(predicted_vals == 1) & (label_mat == 1)] = -0.5
                # err_arr[(predicted_vals == -1) & (label_mat == 1)] = 1
                err_arr[(predicted_vals == 1) & (label_mat == -1)] = 3

                weighted_error = np.dot(weigh_arr.T, err_arr)
                # print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, thresh_val, inequal, weighted_error)
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_class_est = predicted_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal

    return best_stump, min_error, best_class_est


def adaboost_train_ds(data_arr, class_labels, num_it=40, num_ds_steps=10):
    weak_class_arr = []
    m = np.shape(data_arr)[0]
    weigh_arr = np.mat(0.01*np.ones((m, 1))/m)
    agg_class_est = np.mat(np.zeros((m, 1)))

    for i in range(num_it):
        best_stump, error, class_est = build_stump(data_arr, class_labels, weigh_arr, num_ds_steps)
        # print "weigh_arr:", weigh_arr.T

        alpha = float(0.5*np.ma.log((1.0-error)/max(error, 1e-16)))
        best_stump['alpha'] = alpha
        weak_class_arr.append(best_stump)
        # print "class_est: ", class_est.T

        expon = np.ma.multiply(-1*alpha*np.mat(class_labels).T, class_est)
        weigh_arr = np.ma.multiply(weigh_arr, np.ma.exp(expon))
        weigh_arr = weigh_arr/weigh_arr.sum()
        agg_class_est += alpha*class_est
        # print "agg_class_est: ", agg_class_est.T

        agg_errors = np.ma.multiply(sign(agg_class_est) != np.mat(class_labels).T, np.ones((m, 1)))
        error_rate = agg_errors.sum()/m
        print "Iteration {0} ---- total error: {1}".format(i, error_rate)

        if error_rate == 0.0:
            break
    return weak_class_arr, agg_class_est


def adaboost_test_ds(dat_to_class, classifier_arr, thresh=0, ineq='lt'):
    data_matrix = np.mat(dat_to_class)
    m = np.shape(data_matrix)[0]
    agg_class_est = np.mat(np.zeros((m, 1)))
    bin_class_est = np.mat(np.zeros((m, 1)))
    for i in range(len(classifier_arr)):
        class_est = stump_classify(data_matrix, classifier_arr[i]['dim'],
                                   classifier_arr[i]['thresh'],
                                   classifier_arr[i]['ineq'])
        agg_class_est += classifier_arr[i]['alpha']*class_est
        # print agg_class_est

    if ineq == 'lt':
        bin_class_est[agg_class_est <= thresh] = -1
        bin_class_est[agg_class_est > thresh] = 1
    else:
        bin_class_est[agg_class_est > thresh] = -1
        bin_class_est[agg_class_est <= thresh] = 1
    return bin_class_est, agg_class_est