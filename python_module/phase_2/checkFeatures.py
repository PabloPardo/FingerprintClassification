__author__ = 'joan.valls'
from pipeline.utils import *
import cPickle as cPk

test_name = "RandomizedData_Training"
x_path = test_name + "/X_[3]DensRad_[1]GradRad_[5]EntrRad_Hough_32bins.pkl"
path_csv = "//ssd2015/Data/CSVs/RandomizedData.csv"


if os.path.isfile(x_path):
    csv = pd.read_csv(path_csv, delimiter=';')
    names = csv['EmNomFitxer'].values

    with open(x_path, 'rb') as file:
        hist_imgs, _ = cPk.load(file)

    with open(test_name + "/output.csv", 'wb') as f:
        for i in range(len(names)):
            f.write(names[i])
            for j in hist_imgs[i]:
                f.write(',' + str(j))
            f.write('\n')
else:
    print "The model specified doesn't exist."
