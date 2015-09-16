__author__ = 'pablo'

from PIL import Image
import cPickle as cPk
import pandas as pd
import numpy as np
import os.path
import gc
from classification.metrics import confusion_matrix


def read_data(file_name, sheet_num):
    file = open(file_name, 'r')
    df = pd.read_excel(file_name, sheet_num, encoding='utf-8')
    file.close()
    return df


def create_data2(path_csv, path_data, name_data_I, name_data_X, name_data_y, labels, raw_data):

    # Read csv file
    csv = pd.read_csv(path_csv, delimiter=';')

    I = []
    y = []
    X = []
    names = []
    prev = 0
    for i in range(len(csv)):
        name = csv['EmNomFitxer'].values[i]
        if not raw_data :
            try:
                im = Image.open(path_data + name).convert('L')
                if type(I) != list:
                    I = np.dstack((I, im))
                    if prev == I.shape[2]:
                        print '%s -- %d\n' % (name, I.shape[2])
                    prev = I.shape[2]
                else:
                    I = im
            except IOError:
                print 'No Image: ' + name
                continue

        init_y = csv.columns.get_loc('EmBorrosa')
        end_y = init_y + 6
        init_x = csv.columns.get_loc('dedo')
        end_x = csv.columns.get_loc('>.9') + 1
        X.append(csv[csv['EmNomFitxer'] == name].values[0][init_x:end_x])
        y.append(csv[csv['EmNomFitxer'] == name].values[0][init_y:end_y])

        names.append(name)
        print '%d / %d \r' % (i+1, len(csv))
        # gc.collect()

    y = np.array(y, dtype=np.int16)
    X = np.array(X, dtype=np.int32)

    return I, X, y, names

def create_data_cv(path_csv):
    # Read csv file
    csv = pd.read_csv(path_csv, delimiter=';')

    y = []
    names = []
    for i in range(len(csv)):
        name = csv['EmNomFitxer'].values[i]
        y.append(csv[csv['EmNomFitxer'] == name].values[0][4:10])
        names.append(name)
        print '%d / %d \r' % (i, len(csv))

    y = np.array(y, dtype=np.int16)

    return y, names

def create_data(path_xlsx, path_csv, path_data, name_data_I, name_data_X, name_data_y, labels):

    # Read .xlsx file
    # xlsx = read_data(path_xlsx, 3)
    xlsx = read_data(path_xlsx, 0)

    # Read csv file
    csv = pd.read_csv(path_csv, delimiter=';')
    for i in range(len(csv)):
        csv['origen'].values[i] = csv['origen'].values[i].split('.')[0] + '.png'

    # Stack images and get labels
    I = []
    y = []
    X = []
    names = []
    for i in range(len(xlsx)):
        print '%d / %d \r' % (i+1, len(xlsx))
        name = xlsx['EmNomFitxer'].values[i]

        if not os.path.isfile(name_data_I):
            try:
                im = Image.open(path_data + name).convert('L')
            except IOError:
                print 'No Image: ' + name
                continue

        if not os.path.isfile(name_data_X):
            if name in csv['origen'].values:
                X.append(csv[csv['origen'] == name].values[0][2:-4])
            else:
                print 'No csv:   ' + name
                continue

        names.append(name)
        if not os.path.isfile(name_data_I):
            if type(I) != list:
                I = np.dstack((I, im))
            else:
                I = im

        if not os.path.isfile(name_data_y):
            y_ = []
            for l in labels:
                if xlsx[l].values[i] == 1:
                    y_.append(1)
                else:
                    y_.append(0)
            if sum(y_) == 0:
                y_.append(1)
            else:
                y_.append(0)
            y.append(y_)

    if not os.path.isfile(name_data_y):
        y = np.array(y)

    # Save data
    if not os.path.isfile(name_data_I):
        file = open(name_data_I, 'wb')
        cPk.dump(I, file)
        file.close()
    else:
        # Load data
        # I = []
        file = open(name_data_I, 'rb')
        I = cPk.load(file)
        file.close()

    if not os.path.isfile(name_data_X):
        file = open(name_data_X, 'wb')
        cPk.dump(X, file)
        file.close()
    else:
        # X = []
        file = open(name_data_X, 'rb')
        X = cPk.load(file)
        file.close()
    if not os.path.isfile(name_data_y):
        file = open(name_data_y, 'wb')
        cPk.dump(y, file)
        file.close()
    else:
        file = open(name_data_y, 'rb')
        y = cPk.load(file)
        file.close()

    return I, X, y, names


def label_subset(y, subset):
    y_sub = []
    for y_ in y:
        aux = []
        for j in subset:
            aux.append(y_[j])

        if sum(aux) == 0:
            aux.append(1)
        else:
            aux.append(0)
        y_sub.append(aux)
    return np.array(y_sub)


def conf_matrix(y, y_pred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            if y[i] == 1:
                TP += 1
            else:
                TN += 1
        else:
            if y[i] == 1:
                FN += 1
            else:
                FP += 1

    return TP, TN, FP, FN


def write_FP(y, y_pred, file_name, names):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    with open(file_name, 'wb') as f:
        for i in range(len(y)):
            if y[i] == y_pred[i]:
                if y[i] == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if y[i] == 1:
                    FN += 1
                    f.write("FN " + names[i] + '\n')
                else:
                    FP += 1
                    f.write("FP " + names[i] + '\n')

    return TP, TN, FP, FN


def write_results(params, y, output_file, names, name_results_FP, predictions):
    n_bins = params['n_bins']
    n_folds = params['n_folds']
    y_names = params['y_names']
    y_pred = params['y_pred']
    if 'y_pred_prob' in params:
        y_pred_prob = params['y_pred_prob']
    rad_density = params['rad_density']
    rad_gradient = params['rad_gradient']
    rad_entropy = params['rad_entropy']
    estimator_name = params['estimator_name']

    acc = params['InterAcc']
    innAcc = params['InnerAcc']
    WinnAcc = params['WInnerAcc']
    IoU = params['IntOfUni']

    results = {'inter': np.mean(acc),
               'inner': np.mean(innAcc),
               'w_inner': np.mean(WinnAcc),
               'iou': np.mean(IoU)}

    file = open(output_file, 'wb')

    file.write('--- Parameters ---\n')
    file.write('Number of Bins per histogram: %d\n' % n_bins)
    file.write('Radius Density:               %s\n' % str(rad_density))
    file.write('Radius Gradient:              %s\n' % str(rad_gradient))
    file.write('Radius Entropy:               %s\n' % str(rad_entropy))
    file.write('Number of CV folds:           %d\n' % n_folds)
    file.write('Labels names:                 %s\n' % y_names)
    file.write('Estimator name:               %s\n' % estimator_name)

    file.write('\n--- Results ---\n')
    file.write('Inter Class Accuracy:         %f\n' % np.mean(acc))
    file.write('Inner Class Accuracy:         %f\n' % np.mean(innAcc))
    file.write('Weigh Inner Class Accuracy:   %f\n' % np.mean(WinnAcc))
    file.write('Intersection over Union:      %f\n' % np.mean(IoU))

    file.write('\n--- Results per classes ---\n')
    for i in range(len(y_names)):
        TP, TN, FP, FN = write_FP(y[:, i], y_pred[:, i], name_results_FP % y_names[i], names)
        n = len(y)
        results[y_names[i]] = {'TP': TP/float(n),
                               'TN': TN/float(n),
                               'FP': FP/float(n),
                               'FN': FN/float(n)}

        file.write('%s:\n' % y_names[i])
        file.write('\tTrue Positives: %f%%\n' % (TP/float(n)))
        file.write('\tTrue Negative: %f%%\n' % (TN/float(n)))
        file.write('\tFalse Positives: %f%%\n' % (FP/float(n)))
        file.write('\tFalse Negatives: %f%%\n' % (FN/float(n)))
    file.close()

    # Escriure butifarres
    if 'y_pred_prob' in params:
        file = open(predictions, 'wb')
        for i in range(len(y)):
            file.write('%s;%s;%s\n' % (names[i], ';'.join(str(e) for e in y_pred_prob[i]), ';'.join(str(e) for e in y[i])))
        file.close()

    return results


def write_summary(results, params, name):
    f = open(name, 'w')
    for i in results.keys():
        f.write('\n%s\n========\nParameters:\n' % i)
        f.write('\tNumber of bins:  %d\n' % params[i]['n_bins'])
        f.write('\tRad. Density:    %s\n' % str(params[i]['rad_density']))
        f.write('\tRad. Gradient:   %s\n' % str(params[i]['rad_gradient']))
        f.write('\tRad. Entropy:    %s\n' % str(params[i]['rad_entropy']))

        f.write('Results:\n')
        f.write('\t Inter Class Acc.:          %f\n' % results[i]['inter'])
        f.write('\t Inner Class Acc.:          %f\n' % results[i]['inner'])
        f.write('\t Weighted Inner Class Acc.: %f\n' % results[i]['w_inner'])
        f.write('\t Intersection over Union:   %f\n' % results[i]['iou'])

        f.write('Confusion Matrix:\n')
        for j in params[i]['y_names']:
            f.write('\n\t%s:\n' % j)
            f.write('\t\tTrue Positives:  %f\n' % results[i][j]['TP'])
            f.write('\t\tTrue Negatives:  %f\n' % results[i][j]['TN'])
            f.write('\t\tFalse Positives: %f\n' % results[i][j]['FP'])
            f.write('\t\tFalse Negatives: %f\n' % results[i][j]['FN'])
            f.write('\n\t\tAccuracy: %f\n' % (results[i][j]['TP'] + results[i][j]['TN']))
            f.write('\t\tW. Accuracy: %f\n' % (results[i][j]['TP'] / (2.0*(results[i][j]['TP'] + results[i][j]['FP'] + results[i][j]['FN'])) + results[i][j]['TN'] / (2.0*(results[i][j]['TN'] + results[i][j]['FP'] + results[i][j]['FN']))))
    f.close()