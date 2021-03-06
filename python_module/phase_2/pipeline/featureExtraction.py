
__author__ = 'pablo'
from imageProcessing.features import hist_density, hist_grad, hist_entropy, hist_hough, diferentiate_img
from imageProcessing.utils import split_image
from utils import *
import time
from imageProcessing.segmentation import segmentation
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import cPickle as cPk


def getNormDataFromPath(norm_data_path):
    hist_imgs = np.genfromtxt(norm_data_path + '/NormalizedFeatures.csv', delimiter=',')
    hist_imgs = hist_imgs[:, 1:]
    normalization = np.genfromtxt(norm_data_path + '/normalization.csv', delimiter=' ')
    mean = normalization[:, 0]
    std = normalization[:, 1]
    return hist_imgs, mean, std


def feature_extraction(I, X, name_data_x, params, names, verbose=False, test_name='Default', raw_data='RowDefault.pkl', norm_data_path='false', seg_available= False):
    n_bins = params['n_bins']
    rad_density = params['rad_density'] if type(params['rad_density']) == list else [params['rad_density']]
    rad_gradient = params['rad_gradient'] if type(params['rad_gradient']) == list else [params['rad_gradient']]
    rad_entropy = params['rad_entropy'] if type(params['rad_entropy']) == list else [params['rad_entropy']]

    #
    if os.path.isfile(name_data_x % (str(rad_density), str(rad_gradient), str(rad_entropy), n_bins)):
        # Load data
        file = open(name_data_x % (str(rad_density), str(rad_gradient), str(rad_entropy), n_bins), 'rb')
        hist_imgs, params = cPk.load(file)
        file.close()
    else:
        mean = []
        std = []
        if norm_data_path != 'false':
            hist_imgs, mean, std = getNormDataFromPath(norm_data_path)
        else:
            if not raw_data:
                hist_imgs = []
                if verbose:
                    print 'Hole Image\n==========='
                for i in range(I.shape[2]):
                    t = time.time()


                    # Segment image
                    _, seg_image = segmentation(I[:, :, i])
                    if not seg_available:
                        seg_image = I[:, :, i]

                    # Get Histogram of Intensities
                    if type(rad_density) == list:
                        HoI = []
                        for r in rad_density:
                            h = hist_density(seg_image, radius=r, n_bins=n_bins, verbose=verbose)
                            HoI.extend(h)
                    else:
                        HoI = hist_density(seg_image, radius=rad_density, n_bins=n_bins, verbose=verbose)

                    # Get Histogram of Gradients
                    if type(rad_gradient) == list:
                        HoG = []
                        for r in rad_gradient:
                            h = hist_grad(seg_image, radius=r, n_bins=n_bins, verbose=verbose)
                            HoG.extend(h)
                    else:
                        HoG = hist_grad(seg_image, radius=rad_gradient, n_bins=n_bins, verbose=verbose)

                    # Get Histogram of Entropy
                    if type(rad_entropy) == list:
                        HoE = []
                        for r in rad_entropy:
                            h = hist_entropy(seg_image, radius=r, n_bins=n_bins, verbose=verbose)
                            HoE.extend(h)
                    else:
                        HoE = hist_entropy(seg_image, radius=rad_entropy, n_bins=n_bins, verbose=verbose)

                    # Get Histogram of Hough
                    HoH = []
                    h = hist_hough(seg_image, n_bins=n_bins, verbose=verbose)
                    HoH.extend(h)

                    # hist_imgs.append(HoI)
                    hist = np.concatenate((HoI, HoG, HoE, HoH, X[i]))

                    # Split image into small images
                    shape_spl = (3, 2)
                    splt = split_image(seg_image, shape=shape_spl)
                    for j in range(shape_spl[0]*shape_spl[1]):
                        aux = []
                        if verbose:
                            print "Split section %d\n===============" % j
                        for r in rad_density:
                            h = hist_density(splt[j], radius=r, n_bins=n_bins, verbose=verbose)
                            aux.extend(h)
                        for r in rad_gradient:
                            h = hist_grad(splt[j], radius=r, n_bins=n_bins, verbose=verbose)
                            aux.extend(h)
                        for r in rad_entropy:
                            h = hist_entropy(splt[j], radius=r, n_bins=n_bins, verbose=verbose)
                            aux.extend(h)
                        h = hist_hough(splt[j], n_bins=n_bins, verbose=verbose)
                        aux.extend(h)
                        hist = np.concatenate((hist, aux))
                    hist = np.concatenate((hist, diferentiate_img(seg_image, verbose=verbose)))

                    hist_imgs.append(hist)

                    print '%3.2f%% -- time: %f\r' % (i / float(I.shape[2]), time.time() - t)

                with open(test_name + "/features_full_python_unnormalized.csv", 'wb') as f:
                    for i in range(len(hist_imgs)):
                        f.write(names[i])
                        for j in hist_imgs[i]:
                            f.write(',' + str(j))
                        f.write('\n')
                # np.savetxt(test_name + "/features_full_python_unnormalized.csv", csv_data , delimiter=",")
            else:
                if os.path.isfile(raw_data):
                    hist_imgs = np.genfromtxt(raw_data, delimiter=',')
                    #hist_imgs = hist_imgs[:, :]

            # Normalize Features
            #norm_data = np.array(hist_imgs,dtype=np.double)
            hist_imgs = np.array(hist_imgs,dtype=np.double)
            for i in range(hist_imgs.shape[1]):
                if not 'normalize_mean' in params.keys():
                    mean.append(np.mean(hist_imgs[:, i]))
                    std.append(np.std(hist_imgs[:, i]))
                else:
                    mean.append(params['normalize_mean'][i])
                    std.append(params['normalize_std'][i])
                hist_imgs[:, i] = (hist_imgs[:, i] - mean[i]) / std[i] if std[i] != 0 else hist_imgs[:, i] - mean[i]

        params['normalize_mean'] = mean
        params['normalize_std'] = std

        with open(test_name + "/normalization.csv", 'wb') as f:
            for i in range(len(mean)):
                    f.write('%f %f\n' % (mean[i], std[i]))



        # Save data
        file = open(name_data_x % (str(rad_density), str(rad_gradient), str(rad_entropy), n_bins), 'wb')
        cPk.dump((hist_imgs, params), file)
        file.close()

        if norm_data_path == 'false':
            with open(test_name + "/features_full_python.csv", 'wb') as f:
                for i in range(len(hist_imgs)):
                    f.write(names[i])
                    for j in hist_imgs[i]:
                        f.write(',' + str(j))
                    f.write('\n')
    return hist_imgs, params