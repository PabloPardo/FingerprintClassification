from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler

from utils import *
from adaboost import *
from sklearn import svm, cross_validation


# DATA EXTRACTION
# Data to use in the prediction: dedo, nfiq, foregraund, numMinucias, >.1, ..., >.9
data = load('../../data/results_all.csv')


geyce_y = np.array(data.get(data.keys()[-2]).values, dtype=np.int8)  # Predictions given by Geyce
kit4_y = np.array(data.get(data.keys()[-1]).values, dtype=np.int8)   # Labels given by Kit4

# Change Labels to -1, 1 instead of 0, 1
for i in range(len(geyce_y)):
    geyce_y[i] = -1 if geyce_y[i] == 0 else 1
    kit4_y[i] = -1 if kit4_y[i] == 0 else 1

# print len([i for i in kit4_y if not i])/float(len(kit4_y))
# n = len(kit4_y)
data = data.get(data.keys()[3:-3])

# Split the feature 'dedo' into 10 binary features
dedos = data.get('dedo')
dedos = pd.get_dummies(dedos.values, prefix='dedo')
data.drop('dedo', axis=1, inplace=True)
data = data.join(dedos).as_matrix()

# Normalize
X = StandardScaler().fit_transform(data)

# PREDICTION
# Split data into test and train
outer = StratifiedKFold(y=kit4_y, n_folds=10)

fold_cnt = 0
TP = []
TN = []
FP = []
FN = []
TP_geyce = []
TN_geyce = []
FP_geyce = []
FN_geyce = []
for out_train, out_test in outer:
    fold_cnt += 1
    print 'Fold {0} / 10\n-----------\n'.format(fold_cnt)

    X_out_train = X[out_train]
    X_out_test = X[out_test]
    y_out_train = kit4_y[out_train]
    y_out_test = kit4_y[out_test]

    ## TRAIN
    # Grid search2
    # estimator = svm.LinearSVC(dual=False)
    # estimator = svm.SVC()
    # params = {'C': [0.01, 0.1, 0.5, 1, 2, 3]}
    #           #'gamma': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]}
    # grid = GridSearchCV(estimator=estimator, param_grid=params, n_jobs=4, cv=10, verbose=0)
    # grid.fit(X_out_train, y_out_train)
    weak_class_arr, agg_class_est_train = adaboost_train_ds(X_out_train, y_out_train, num_it=40, num_ds_steps=10)

    ## VALIDATE
    # y_out_test_pred = grid.predict(X_out_test)
    y_out_test_pred, agg_class_est_test = adaboost_test_ds(X_out_test, weak_class_arr, thresh=1.5)

    ## EVALUATE
    contingency_table = eval_pred(y_out_test_pred, y_out_test)

    TP.append(contingency_table[0])
    TN.append(contingency_table[1])
    FP.append(contingency_table[2])
    FN.append(contingency_table[3])
    # print '\tBest param: {0}\n'.format(grid.best_params_)

    # --------------------------------------------
    print '\nGeyce\'s Evaluation\n------------------\n'
    contingency_table_geyce = eval_pred(geyce_y[out_test], y_out_test)
    TP_geyce.append(contingency_table_geyce[0])
    TN_geyce.append(contingency_table_geyce[1])
    FP_geyce.append(contingency_table_geyce[2])
    FN_geyce.append(contingency_table_geyce[3])

    plotFP(agg_class_est_test, y_out_test, [0, 3], 80)
    plotROC(agg_class_est_test.T, y_out_test)
    break

print 'Our Prediction Stats\n--------------------'
print '\tTP = %f' % np.mean(TP)
print '\tTN = %f' % np.mean(TN)
print '\tFP = %f' % np.mean(FP)
print '\tFN = %f' % np.mean(FN)

print 'Geyces Prediction Stats\n-----------------------'
print '\tTP = %f' % np.mean(TP_geyce)
print '\tTN = %f' % np.mean(TN_geyce)
print '\tFP = %f' % np.mean(FP_geyce)
print '\tFN = %f' % np.mean(FN_geyce)
