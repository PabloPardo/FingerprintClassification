__author__ = 'pablo'
import numpy as np
from sklearn.metrics import accuracy_score


def intersection_bin(a, b):
    i = 0
    for j in range(len(a)):
        if a[j] == b[j] == 1:
            i += 1
    return i


def union_bin(a, b):
    u = 0
    for i in range(len(a)):
        if a[i] == 1:
            u += 1
        else:
            if b[i] == 1:
                u += 1
    return u


def intersection_over_union_score(y, y_pred):
    a = []
    for i in range(len(y)):
        I = intersection_bin(y[i], y_pred[i])
        U = union_bin(y[i], y_pred[i])
        a.append(I / float(U))

    return np.mean(a)


def confusion_matrix(y, y_pred):
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


def weigh_accuracy_score(y, y_pred):
    TP, TN, FP, FN = confusion_matrix(y, y_pred)
    p1 = TP / (2.0*(TP + FP + FN)) if TP + FP + FN != 0 else .5
    p2 = TN / (2.0*(TN + FP + FN)) if TN + FP + FN != 0 else .5
    return p1 + p2


def weigh_inner_class_acc_score(y, y_pred):
    m, n = y.shape
    a = []
    for i in range(n):
        a.append(weigh_accuracy_score(y[:, i], y_pred[:, i]))

    return np.mean(a)


def inner_class_acc_score(y, y_pred):
    m, n = y.shape
    a = []
    for i in range(n):
        a.append(accuracy_score(y[:, i], y_pred[:, i]))

    return np.mean(a)