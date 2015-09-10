import os
from PIL import Image
import numpy as np
from imageProcessing.features import hist_density, hist_grad, hist_entropy, hist_hough, diferentiate_img
from imageProcessing.utils import split_image
from scipy.misc import imsave


def feature_extraction(I, params, verbose=False):
    n_bins = params['n_bins']
    rad_density = params['rad_density'] if type(params['rad_density']) == list else [params['rad_density']]
    rad_gradient = params['rad_gradient'] if type(params['rad_gradient']) == list else [params['rad_gradient']]
    rad_entropy = params['rad_entropy'] if type(params['rad_entropy']) == list else [params['rad_entropy']]

    hist_imgs = []
    if verbose:
        print 'Hole Image\n==========='

    # Segment image
    # _, seg_image = segmentation(I[:, :, i])
    seg_image = np.array(I)

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

    return hist_imgs, params


def load_vect(path):
    with open(path, 'rb') as file:
        vect = file.readlines()
    return [float(i) for i in vect[0].split(',')]


def vect_comp(v1, v2):
    return sum([np.abs(v1[i] - v2[i]) for i in range(len(v1))]) / float(len(v1))


def vect_norm(v):
    return (v - min(v)) / (max(v) - min(v))


def main(dir, image_path, comp_data, feature):
    """
        ----------------------------
                Load Data
                ---------
        ----------------------------
    """
    n_bins = 32
    rad_density = 3
    rad_gradient = 1
    rad_entropy = 5
    verbose = False

    params = {
        'n_classes': '6Class',
        'n_jobs': -1,
        'n_bins': 32,
        'rad_density': 3,
        'rad_gradient': 1,
        'rad_entropy': 5
    }

    I = np.array(Image.open(dir + image_path).convert('L'))

    if feature == 'd' or feature == 'd_img':
        h, feat_img = hist_density(I, radius=rad_density, n_bins=n_bins, verbose=verbose)
    elif feature == 'g' or feature == 'g_img':
        h, feat_img = hist_grad(I, radius=rad_gradient, n_bins=n_bins, verbose=verbose)
    elif feature == 'e' or feature == 'e_img':
        h, feat_img = hist_entropy(I, radius=rad_entropy, n_bins=n_bins, verbose=verbose)
    elif feature == 'h' or feature == 'h_img':
        h, feat_img = hist_hough(I, n_bins=n_bins, verbose=verbose)
    else:
        print 'Wrong feature.'
        return -1

    v1 = np.array(load_vect(dir + comp_data))

    if feature == 'd_img' or feature == 'g_img' or feature == 'e_img' or feature == 'h_img':
        v2 = feat_img.flatten()
    else:
        v2 = h

    error = vect_comp(v1, v2)
    print error

    if feature == 'd_img' or feature == 'g_img' or feature == 'e_img' or feature == 'h_img':
        v1 = vect_norm(v1).reshape([512, 512])
        v2 = vect_norm(v2).reshape([512, 512])

        imsave(dir + 'ImageC++_%s.jpg' % feature, v1)
        imsave(dir + 'ImagePyt_%s.jpg' % feature, v2)

    # hist_imgs, params = feature_extraction(I, params, verbose=False)

    return 0

if __name__ == "__main__":
    path = 'TestFeatures/'
    img_name = '2014-11-26_7938306_06.png'
    test_dir = path + img_name.split('.')[0] + '/'

    if not os.path.isdir(test_dir):
        os.makedirs(test_dir)

    main(test_dir, img_name, 'edges2014-11-26_7938306_06NOEOF.TXT', 'h_img')
