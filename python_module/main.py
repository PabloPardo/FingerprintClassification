__author__ = 'pablo'

import time
from plot import plot_scores
from pipeline.utils import *
from itertools import product
from pipeline.learningMethod import train, train_testCV
from sklearn.ensemble import RandomForestClassifier
from pipeline.featureExtraction import feature_extraction
import cPickle as cPk
"""
    ----------------------------
            Load Data
            ---------
    ----------------------------
"""

# Definition of all the I/O paths
# size = 'big'
test_name = 'RandomizedData_Training2.0'
path_data = '//ssd2015/Data/Training/'
path_xlsx = '//ssd2015/Data/CSVs/RandomizedData.csv'
# path_csv = path_data + 'results_all.csv'
name_data_img = test_name + '/stack_images.pkl'
name_data_y = test_name + '/y.pkl'
name_data_x = test_name + '/X_%sDensRad_%sGradRad_%sEntrRad_Hough_%dbins.pkl'
name_data_y_pred = test_name + '/model.pkl'
name_data_geyce_x = test_name + '/X.pkl'
name_results = test_name + '/results.txt'
name_results_FP = test_name + '/FP_res_%s_Class.txt'
raw_data = False

labels = [ 'EmBorrosa', 'EmPetita', 'EmNegre', 'EmClara',  'EmMotejada', 'EmDefectuosa']

if not os.path.exists(test_name):
        os.makedirs(test_name)

I, X, y, names = create_data2(path_xlsx, path_data, name_data_img,name_data_geyce_x, name_data_y, labels, raw_data)

# Create matrix of subset of labels we want to classify the instances on
classes = {'6Classes': (['Blurry', 'Small', 'Dark', 'Bright',  'Spotted', 'Damaged'], [0, 1, 2, 3, 4, 5])}
"""
    -------------------------------------
            Feature Extraction
            ------------------
    -------------------------------------
"""
# Define parameters for the feature extraction.
# n_bins and rad_* can be a list of elements, then the program will check the
# performance of different parameter configurations.
num_classes = ['6Classes']
n_bins = 32
rad_density = 3
rad_entropy = 5
max_depth = None
aux = [[n_bins, rad_density, rad_entropy, i] for i in num_classes]
best_results = {'6Classes': {'inter': 0, 'inner': 0, 'w_inner': 0, 'iou': 0}}
best_params = {}
counter = 0

for i in aux:
    tinit = time.time()
    # y = label_subset(y_, classes[i[3]][1])
    params = {'y_names': classes[i[3]][0],
              'n_classes': i[3],
              'n_jobs': -1,
              'n_bins': i[0],
              'rad_density': i[1],
              'rad_gradient': 1,
              'rad_entropy': i[2]}

    hist_imgs, params = feature_extraction(I, X, name_data_x, params, names, False, test_name, raw_data)

    # -- Use C++ features --
    # hist_imgs = np.genfromtxt('//ssd2015/Data/CSVs/FitFeaturesGEYCE.csv', delimiter=',')
    # hist_imgs = hist_imgs[:, 1:]
    # mean = []
    # std = []
    # hist_imgs = np.array(hist_imgs)
    # for i in range(hist_imgs.shape[1]):
    #     mean.append(np.mean(hist_imgs[:, i]))
    #     std.append(np.std(hist_imgs[:, i]))
    #     hist_imgs[:, i] = (hist_imgs[:, i] - mean[i]) / std[i] if std[i] != 0 else hist_imgs[:, i] - mean[i]
    #
    # params['normalize_mean'] = mean
    # params['normalize_std'] = std



    """
        --------------------
                Train
                -----
        --------------------
    """
    params['estimator'] = RandomForestClassifier()
    params['estimator_name'] = 'RandomForest'
    params['estimator_params'] = {'n_estimators': 10, 'max_depth': max_depth}
    params['n_folds'] = 5
    params['verbose'] = 0

    if os.path.isfile(name_data_y_pred):
        file = open(name_data_y_pred, 'rb')
        params = cPk.load(file)
        file.close()
    else:
        params = train_testCV(hist_imgs, y, params, name_data_y_pred)

    r = write_results(params, y, name_results, names, name_results_FP)

    if best_results[params['n_classes']]['inner'] < r['inner']:
        best_results[params['n_classes']] = r
        best_params[params['n_classes']] = params

    counter += 1
    print '%.2f%% : params: %s - time: %f\r' % (100 * counter / float(len(aux)), str(i), time.time() - tinit)

write_summary(best_results, best_params, test_name + '/summary.txt')
