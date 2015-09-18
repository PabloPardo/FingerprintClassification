__author__ = 'pablo'
import pandas as pd
import numpy as np
import time
from PIL import Image
from segmentation import segmentation
from features import hist_hough, hist_entropy, hist_grad, hist_density, diferentiate_img, split_image


def load(path):
    """
    Loads the data from the results.csv

    :param path: Path to the csv file
    :type path: str

    :rtype: pd.DataFrame
    :return: Returns a pandas DataFrame.
    """
    data = pd.read_csv(path, comment='#', delimiter=';', index_col=False)

    return data


def load_extended(path, param):
    """
    Loads the data from the results.csv and computes extended features.

    :param path: Path to the csv file
    :type path: str

    :param param: Dictionary with the parameters necessaries to compute the features.
    :type param: dict

    :rtype: pd.DataFrame
    :return: Returns a pandas DataFrame.
    """
    data = pd.read_csv(path, comment='#', delimiter=';', index_col=False)

    rad_density = param['rad_density']
    rad_gradient = param['rad_gradient']
    rad_entropy = param['rad_entropy']
    n_bins = param['n_bins']

    hist_imgs = []
    for i in range(len(data.index)):
        t = time.time()

        I = Image.open(path + data['archivo'].values[i]).convert('L')

        # Segment image
        _, seg_image = segmentation(I)

        # Get Histogram of Intensities
        if type(rad_density) == list:
            HoI = []
            for r in rad_density:
                h = hist_density(seg_image, radius=r, n_bins=n_bins)
                HoI.extend(h)
        else:
            HoI = hist_density(seg_image, radius=rad_density, n_bins=n_bins)

        # Get Histogram of Gradients
        if type(rad_gradient) == list:
            HoG = []
            for r in rad_gradient:
                h = hist_grad(seg_image, radius=r, n_bins=n_bins)
                HoG.extend(h)
        else:
            HoG = hist_grad(seg_image, radius=rad_gradient, n_bins=n_bins)

        # Get Histogram of Entropy
        if type(rad_entropy) == list:
            HoE = []
            for r in rad_entropy:
                h = hist_entropy(seg_image, radius=r, n_bins=n_bins)
                HoE.extend(h)
        else:
            HoE = hist_entropy(seg_image, radius=rad_entropy, n_bins=n_bins)

        # Get Histogram of Hough
        HoH = []
        h = hist_hough(seg_image, n_bins=n_bins)
        HoH.extend(h)

        # hist_imgs.append(HoI)
        hist = np.concatenate((HoI, HoG, HoE, HoH))

        # Split image into small images
        shape_spl = (3, 2)
        splt = split_image(seg_image, shape=shape_spl)
        for j in range(shape_spl[0]*shape_spl[1]):
            aux = []
            for r in rad_density:
                h = hist_density(splt[j], radius=r, n_bins=n_bins)
                aux.extend(h)
            for r in rad_gradient:
                h = hist_grad(splt[j], radius=r, n_bins=n_bins)
                aux.extend(h)
            for r in rad_entropy:
                h = hist_entropy(splt[j], radius=r, n_bins=n_bins)
                aux.extend(h)
            h = hist_hough(splt[j], n_bins=n_bins)
            aux.extend(h)
            hist = np.concatenate((hist, aux))

        hist = np.concatenate((hist, diferentiate_img(seg_image)))
        hist_imgs.append(hist)

        print '%d / %d -- time: %f' % (i, I.shape[2], time.time() - t)
    df = pd.DataFrame(hist_imgs)

    return pd.concat([data, df], axis=1)


def eval_pred(pred_y, y, verbose=True):
    """
    Evaluates a prediction with the real labels and return
    a list of proportion of true positives, true negatives,
    false positives and false negatives.

    :type pred_y: npumpy.ndarray of int8
    :param pred_y: Predicted binary labels
    :type y: npumpy.ndarray of int8
    :param y: Real binary labels

    :rtype: npumpy.ndarray
    :return: [true_pos, true_neg, false_pos, false_neg]
    """
    assert pred_y.shape[0] == len(y), 'The legth of both label sets must be equal.'
    n = len(y)

    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    for i in range(n):
        if y[i] == pred_y[i]:
            if y[i] > 0:
                true_pos += 1
            else:
                true_neg += 1
        else:
            if pred_y[i] > 0:
                false_pos += 1
            else:
                false_neg += 1

    if verbose:
        acc = (true_pos + true_neg)/float(n)
        err = (false_pos + false_neg)/float(n)
        NPV = true_neg/float(true_neg + false_neg + 1e-16)
        FPR = false_pos/float(false_pos + true_neg + 1e-16)
        pre = true_pos/float(true_pos + true_neg + 1e-16)
        rec = true_pos/float(true_pos + false_neg + 1e-16)
        f1 = 2*pre*rec/(pre+rec)
        print '\tAccuracy: {0}\n\tError Rate: {1}\n\tPrecision: ' \
              '{2}\n\tRecall: {3}\n\tF1 Score: {4}\n\tNPV: {5}\n\tFPR: {6}\n' \
              '\tTP: {7}\n\tTN: {8}\n\tFP: {9}\n\tFN: {10}\n'.format(acc, err, pre, rec, f1,
                                                                      NPV, FPR, true_pos, true_neg,
                                                                      false_pos, false_neg)
    return np.array([true_pos, true_neg, false_pos, false_neg])


def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)
    ySum = 0.0
    numPosClas = sum(np.array(classLabels) == 1.0)
    yStep = 1/float(numPosClas)
    xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
        ySum += cur[1]
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
        cur = (cur[0]-delX, cur[1]-delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print "the Area Under the Curve is: ", ySum*xStep


def plotFP_(agg_class_pred, class_labels, interval, num_step, ineq='lt'):
    step_sz = abs(interval[0] - interval[1]) / float(num_step)
    step = [interval[0] + i*step_sz for i in range(num_step)]
    bin_class_est = np.mat(np.zeros((len(agg_class_pred), 1)))
    TP = []
    TN = []
    FP = []
    FN = []
    for i in step:
        if ineq == 'lt':
            bin_class_est[agg_class_pred <= i] = -1
            bin_class_est[agg_class_pred > i] = 1
        else:
            bin_class_est[agg_class_pred > i] = -1
            bin_class_est[agg_class_pred <= i] = 1

        c = eval_pred(bin_class_est, class_labels, verbose=False)
        TP.append(c[0])
        TN.append(c[1])
        FP.append(c[2])
        FN.append(c[3])
    return TP, TN, FP, FN


def plotFP(agg_class_pred, class_labels, interval, num_step, ineq='lt'):
    step_sz = abs(interval[0] - interval[1]) / float(num_step)
    step = [interval[0] + i*step_sz for i in range(num_step)]
    bin_class_est = np.mat(np.zeros((len(agg_class_pred), 1)))
    TP = []
    TN = []
    FP = []
    FN = []
    for i in step:
        if ineq == 'lt':
            bin_class_est[agg_class_pred <= i] = -1
            bin_class_est[agg_class_pred > i] = 1
        else:
            bin_class_est[agg_class_pred > i] = -1
            bin_class_est[agg_class_pred <= i] = 1

        c = eval_pred(bin_class_est, class_labels, verbose=False)
        TP.append(c[0])
        TN.append(c[1])
        FP.append(c[2])
        FN.append(c[3])

    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots()
    ax1.plot(step, FP, 'b-')
    ax1.set_xlabel('Threshold')
    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel('False Positives', color='b')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')


    ax2 = ax1.twinx()
    ax2.plot(step, FN, 'r-')
    ax2.set_ylabel('False Negative', color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    plt.show()


