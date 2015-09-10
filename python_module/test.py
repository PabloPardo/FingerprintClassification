from classification.metrics import weigh_inner_class_acc_score, intersection_over_union_score, inner_class_acc_score
from pipeline.utils import *
from pipeline.featureExtraction import feature_extraction
import matplotlib.pyplot as plt
import cPickle as cPk


def OneLabelTest(y, prediction, threshold):
    # One Label Test
    one_label_pred = []
    one_label_ref = []
    one_label_class_true = []
    one_label_class_false = []
    one_label_class_not_confidence_true = []
    one_label_class_not_confidence_false = []
    for i in range(len(prediction)):
        max_prob = 0
        max_class = None
        for j in range(len(prediction[i])):
            if prediction[i, j] > max_prob:
                max_prob = prediction[i, j]
                max_class = j
        one_label_pred.append(1 if prediction[i, max_class] > threshold else 0)
        one_label_ref.append(y[i, max_class])
        if one_label_pred[i] == 1:
            if one_label_pred[i] == one_label_ref[i]:
                one_label_class_true.append(max_class)
            else:
                one_label_class_false.append(max_class)
        else:
            if one_label_pred[i] != one_label_ref[i]:
                one_label_class_not_confidence_false.append(max_class)
            else:
                one_label_class_not_confidence_true.append(max_class)

    one_label_acc = 100 * sum([1 if one_label_pred[i] == one_label_ref[i] else 0 for i in range(len(one_label_pred))]) / float(len(one_label_ref))

    return one_label_acc, \
           (
               one_label_class_true,
               one_label_class_false,
               one_label_class_not_confidence_true,
               one_label_class_not_confidence_false
           )


def successVsFailure(conf_mat):
    return (
        (conf_mat[0] + conf_mat[1]) / float(sum(conf_mat)),
        (conf_mat[2] + conf_mat[3]) / float(sum(conf_mat))
    )


def ThresholdTest(y, prediction):
    conf_mat_classes = {i: [] for i in range(len(y[0]))}
    for thr in np.array(range(1, 11)) / 10.0:  # Try thresholds 0.1, 0.2, ... , 0.9, 1
        y_pred = np.array([[1 if j > thr else 0 for j in i] for i in prediction])

        for i in range(len(y[0])):
            conf_mat_classes[i].append(successVsFailure(conf_matrix(y[:, i], y_pred[:, i])))  # [TP, TN, FP, FN]

    for key in conf_mat_classes.keys():
        thr = np.array(range(1, 11)) / 10.0
        success = [i[0] for i in conf_mat_classes[key]]
        failure = [i[1] for i in conf_mat_classes[key]]

        plt.figure()
        plt.plot(thr, success, color='b')
        plt.plot(thr, failure, color='r')
        plt.show()
        # plt.savefig('name.png', 'png')


def main(trained_model_path, labels_path, images_path, features_path, test_name, raw_data):

    """
        ----------------------------
                Load Data
                ---------
        ----------------------------
    """

    # Definition of all the I/O paths
    name_data_img = test_name + '/stack_images.pkl'
    name_data_y = test_name + '/y.pkl'
    name_data_x = test_name + '/X_%sDensRad_%sGradRad_%sEntrRad_Hough_%dbins.pkl'
    name_data_names = test_name + '/arrayNames.pkl'
    name_data_geyce_x = test_name + '/X.pkl'

    """
        --------------------------------
                Load Trained Model
                ------------------
        --------------------------------
    """
    if os.path.isfile(trained_model_path):
        file = open(trained_model_path, 'rb')
        params = cPk.load(file)
        file.close()
    else:
        print "The model specified doesn't exist."
        return 0

    """
        --------------------------------
                Create Test Data
                ----------------
        --------------------------------
    """

    I, X, y, names = create_data2(path_csv=labels_path,path_data=images_path,name_data_I=name_data_img, name_data_X=name_data_geyce_x, name_data_y=name_data_y,name_data_names=name_data_names,labels=params['y_names'],raw_data=raw_data)

    if os.path.isfile(features_path):
        file = open(features_path, 'rb')
        hist_imgs, params = cPk.load(file)
        file.close()
    else:
        hist_imgs, params = feature_extraction(I, X, name_data_x, params,names, False,  test_name, raw_data)

    params['y_names'] = ['EmBorrosa', 'EmPetita', 'EmNegre', 'EmClara', 'EmMotejada', 'EmDefectuosa']

    """
        --------------------------------
                      Test
                      ----
        --------------------------------
    """

    params['y_pred'] = params['bestEstimator'].predict(hist_imgs)
    params['y_pred_prob'] = params['bestEstimator'].predict_proba(hist_imgs)

    # Parche
    # y = list(y)
    # for i in range(len(y)):
    #     y[i] = list(y[i])
    #     y[i].append(0)

    params['InterAcc'] = params['bestEstimator'].score(X=hist_imgs, y=np.array(y))
    params['InnerAcc'] = inner_class_acc_score(np.array(y), params['y_pred'])
    params['WInnerAcc'] = weigh_inner_class_acc_score(np.array(y), params['y_pred'])
    params['IntOfUni'] = intersection_over_union_score(np.array(y), params['y_pred'])

    write_results(params,y,test_name + "/sortida.txt", names, test_name + "/res_FP_%s.txt", test_name + '/predic.txt')

    print "\nSCORES\n======"
    print "Accuracy\t%f" % params['InterAcc']
    print "Inner Accuracy\t%f" % params['InnerAcc']
    print "Weighted Accuracy\t%f" % params['WInnerAcc']
    print "Intersection over Union\t%f" % params['IntOfUni']

    one_label_acc, one_label_class = OneLabelTest(y=y, prediction=params['y_pred_prob'], threshold=.5)
    print '\nOne Label Scores\n================'
    print 'Above Threshold:'
    print '\tTrue : %s' % np.histogram(one_label_class[0], bins=6)[0].__str__()
    print '\tFalse: %s' % np.histogram(one_label_class[1], bins=6)[0].__str__()
    print 'Below Threshold:'
    print '\tTrue : %s' % np.histogram(one_label_class[2], bins=6)[0].__str__()
    print '\tFalse: %s' % np.histogram(one_label_class[3], bins=6)[0].__str__()

    print '\nAccuracy: %2.2f%%' % one_label_acc

    ThresholdTest(y, params['y_pred_prob'])

    return 0

if __name__ == "__main__":
    test_name = "Malos_15_07_08"
    model_path = "RandomizedData_Training/model.pkl"
    labels_path = "//ssd2015/Data/CSVs/"+test_name+".csv"
    images_path = "//ssd2015/Data/PredictData/"
    raw_data = test_name + "/features_full_python_unnormalized.csv"
    #raw_data = False
    #features_path = "data/predict/features_python100.pkl"
    features_path = "eof"
    if not os.path.exists(test_name):
        os.makedirs(test_name)
    main(model_path, labels_path, images_path, features_path, test_name, raw_data)
