from sklearn.ensemble import RandomForestClassifier

__author__ = 'pablo'
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from pipeline.utils import *
import cPickle as Cpk
import copy
from classification.metrics import intersection_over_union_score, inner_class_acc_score, weigh_inner_class_acc_score


def train(X, y, params, output_file):
    n_bins = params['n_bins']
    n_jobs = params['n_jobs']
    verbose = params['verbose']

    rad_density = params['rad_density']
    rad_gradient = params['rad_gradient']
    rad_entropy = params['rad_entropy']

    estimator = params['estimator']
    estimator_name = params['estimator_name']
    estimator_params = params['estimator_params']

    clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=estimator_params['n_estimators'],
                                                     n_jobs=n_jobs,
                                                     max_depth=estimator_params['max_depth'],
                                                     verbose=verbose))
    clf.fit(X, y)

    params['bestEstimator'] = clf

    #file = open(output_file % (params['n_classes'], str(rad_density), str(rad_gradient), str(rad_entropy), n_bins, estimator_name), 'wb')
    file = open(output_file, 'wb')
    cPk.dump(params, file)
    file.close()

    return params


def train_testCV(X, y, params, output_file):
    n_bins = params['n_bins']
    n_jobs = params['n_jobs']
    n_folds = params['n_folds']
    verbose = params['verbose']

    rad_density = params['rad_density']
    rad_gradient = params['rad_gradient']
    rad_entropy = params['rad_entropy']

    estimator = params['estimator']
    estimator_name = params['estimator_name']
    estimator_params = params['estimator_params']

    kf = KFold(len(X), n_folds=n_folds)

    acc = []
    w_acc = []
    IoU = []
    innAcc = []
    y_pred = []
    best_InnerAcc = 0
    for train, test in kf:
        train_x = np.array(X)[train]
        test_x = np.array(X)[test]
        train_y = y[train]
        test_y = y[test]

        clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=estimator_params['n_estimators'],
                                                         n_jobs=n_jobs,
                                                         max_depth=estimator_params['max_depth'],
                                                         verbose=verbose))

        # clf = OneVsRestClassifier(GridSearchCV(estimator=estimator,
        #                                        param_grid=estimator_params,
        #                                        n_jobs=n_jobs,
        #                                        cv=n_folds,
        #                                        verbose=verbose))
        clf.fit(train_x, train_y)

        """
            --------------------
                    Test
                    -----
            --------------------
        """
        y_ = clf.predict(test_x)
        y_pred.extend(y_)

        acc.append(clf.score(test_x, test_y))
        w_acc.append(weigh_inner_class_acc_score(test_y, y_))
        IoU.append(intersection_over_union_score(test_y, y_))
        innAcc.append(inner_class_acc_score(test_y, y_))

        if best_InnerAcc < innAcc[-1]:
            best_InnerAcc = innAcc[-1]
            best_estimator = copy.deepcopy(clf)

    params['InterAcc'] = acc
    params['WInnerAcc'] = w_acc
    params['InnerAcc'] = innAcc
    params['IntOfUni'] = IoU
    params['bestEstimator'] = best_estimator
    params['y_pred'] = np.array(y_pred)
    print best_estimator

    file = open(output_file, 'wb')
    cPk.dump(params, file)
    file.close()

    return params