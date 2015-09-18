__author__ = 'joan.valls'
from pipeline.utils import *
import cPickle as cPk

test_name = "RandomizedData_Training"
trained_model_path = test_name + "/model.pkl"

if os.path.isfile(trained_model_path):
    with open(trained_model_path, 'rb') as file:
        params = cPk.load(file)
    mean = params['normalize_mean']
    std = params['normalize_std']

    with open(test_name  + '/normalization.csv', 'wb') as file:
        for i in range(len(mean)):
            file.write(' '.join((str(mean[i]), str(std[i]))) + '\n')
else:
    print "The model specified doesn't exist."

