__author__ = 'pablo'
import sys
import numpy as np
from skimage.morphology import disk
from skimage.filters.rank import entropy
from math import pi
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line


def integral_image(x):
    """Integral image / summed area table.

    The integral image contains the sum of all elements above and to the
    left of it, i.e.:

    .. math::

       S[m, n] = \sum_{i \leq m} \sum_{j \leq n} X[i, j]

    Parameters
    ----------
    x : ndarray
        Input image.

    Returns
    -------
    S : ndarray
        Integral image / summed area table.

    References
    ----------
    .. [1] F.C. Crow, "Summed-area tables for texture mapping,"
           ACM SIGGRAPH Computer Graphics, vol. 18, 1984, pp. 207-212.

    """
    return x.cumsum(1).cumsum(0)


def integrate(ii, x0, y0, x1, y1):
    """Use an integral image to integrate over a given window.

    Parameters
    ----------
    ii : ndarray
        Integral image.
    x0, y0 : int
        Top-left corner of block to be summed.
    x1, y1 : int
        Bottom-right corner of block to be summed.

    Returns
    -------
    S : int
        Integral (sum) over the given window.

    """
    S = 0

    S += ii[x1, y1]

    if (x0 - 1 >= 0) and (y0 - 1 >= 0):
        S += ii[x0 - 1, y0 - 1]

    if x0 - 1 >= 0:
        S -= ii[x0 - 1, y1]

    if y0 - 1 >= 0:
        S -= ii[x1, y0 - 1]

    return S


def density_pix(i, j, ii, sz_area):
    """ Computes the density around the pixel (i,j).

    :param i: First coordinate of the pixel.
    :type i: int

    :param j: Second coordinate of the pixel.
    :type j: int

    :param ii: Integral image.
    :type ii: ndarray

    :param sz_area: Number of surrounding neighbours to compute density.
    :type sz_area: int

    :return: int
    """
    m, n = ii.shape
    x0 = max(0, i - sz_area)
    y0 = max(0, j - sz_area)
    x1 = min(m - 1, i + sz_area)
    y1 = min(n - 1, j + sz_area)
    return 255 - integrate(ii, x0, y0, x1, y1) / float((x1-x0)*(y1-y0))


def density_img(img, radius, verbose=False):
    """
    This function computes the density image of an image.
    Computing the intensity of a pixel in base of the average
    of its surrounding pixels.

    :param img: Image to use.
    :type img: ndarray

    :param radius: Radius of the neighbours to take into account.
    :type radius: int

    :return: ndarray
    """
    m, n = img.shape
    ii = integral_image(np.asarray(img))

    dens_img = np.empty((m, n))
    for i in range(m):
        x0 = max(0, i - radius)
        x1 = min(m - 1, i + radius)
        for j in range(n):
            y0 = max(0, j - radius)
            y1 = min(n - 1, j + radius)
            if verbose:
                sys.stdout.write('Integrate Pixel (%d, %d): %f\n' % (i, j, integrate(ii, x0, y0, x1, y1)))
            dens_img[i, j] = 255 - integrate(ii, x0, y0, x1, y1) / float((x1-x0)*(y1-y0))

    if verbose:
        sys.stdout.write('Integral Image:\n')
        sys.stdout.write(ii.__str__())

        sys.stdout.write('Density Image:\n')
        sys.stdout.write(dens_img.__str__())

    return dens_img


def hist_density(img, radius, n_bins=64, verbose=False):
    """
    Computes the histogram of densities from an image.

    :param img: Image to be precessed.
    :type img: ndarray

    :param radius: Radius to use for the density calculation.
    :type radius: int

    :param n_bins: Number of bins to compute the histogram.
    :type n_bins: int

    :return: list of int
    """
    dens_img = density_img(img, radius, verbose=verbose)
    hist, _ = np.histogram(dens_img.flatten(), bins=n_bins)

    if verbose:
        sys.stdout.write('\nHistogram of Densities:\n')
        sys.stdout.write(repr(hist.__str__()))

    return hist, dens_img


def gradient_pix(i, j, img, radius):
    """
    Computes the gradient in a pixel using the pixels at a certain
    radius.

    :param i: Pixel row coordinate.
     :type i: int

    :param j: Pixel column coordinate.
     :type j: int

    :param img: Image.
     :type img: ndarray

    :param radius: Radius to us in the calculation.
     :type radius: int

    :return: Gradient Vector
    """
    di = (float(img[i+radius, j]) - float(img[i-radius, j])) / float(2*radius)
    dj = (float(img[i, j+radius]) - float(img[i, j-radius])) / float(2*radius)
    return [di, dj]


def gradient_img(img, radius, verbose=False):
    """
    Computes the image of gradient magnitudes.

    :param img: Image.
     :type img: ndarray

    :param radius: Radius for the gradient calculation.
     :type radius: int

    :return: Gradient Image
    """
    m, n = img.shape
    grad_img = np.empty((m - 2*radius, n - 2*radius))
    for i in range(radius, m-radius):
        for j in range(radius, n-radius):
            grad_img[i-radius, j-radius] = np.linalg.norm(gradient_pix(i, j, img, radius))

    if verbose:
        sys.stdout.write('\nImage of Gradients:\n')
        sys.stdout.write(grad_img.__str__())

    return grad_img


def hist_grad(img, radius=1, n_bins=64, verbose=False):
    """
    Computes the histogram of gradients from an image.

    :param img: Image to be precessed.
    :type img: ndarray

    :param radius: Radius to use for the gradient calculation.
    :type radius: int

    :param n_bins: Number of bins to compute the histogram.
    :type n_bins: int

    :return: list of int
    """
    grad_img = gradient_img(img, radius, verbose=verbose)
    hist, _ = np.histogram(grad_img.flatten(), bins=n_bins)

    if verbose:
        sys.stdout.write('\nHistogram of Gradients:\n')
        sys.stdout.write(hist.__str__())

    return hist, grad_img


def hist_entropy(img, radius=5, n_bins=64, verbose=False):
    """
    Computes the histogram of entropies from an image.

    :param img: Image to be precessed.
    :type img: ndarray

    :param radius: Radius to use for the entropy calculation.
    :type radius: int

    :param n_bins: Number of bins to compute the histogram.
    :type n_bins: int

    :return: list of int
    """
    # entropy_img = entropy(img, disk(radius))
    entropy_img = entropy(img, np.ones([11, 11], dtype=int))
    if verbose:
        sys.stdout.write('\nImage of Entropies:\n')
        sys.stdout.write(entropy_img.__str__())

    hist, _ = np.histogram(entropy_img.flatten(), bins=n_bins)

    if verbose:
        sys.stdout.write('\nHistogram of Entropy:\n')
        sys.stdout.write(hist.__str__())

    return hist, entropy_img


def diferentiate_img(img, verbose=False):
    m, n = img.shape

    row_split = m / 3
    col_split = n / 2

    ii = integral_image(img)

    dif = []
    for i in range(3):
        x0 = i*row_split
        y0 = 0
        x1 = (i+1)*row_split-1
        y1 = col_split

        x2 = x0
        y2 = col_split
        x3 = x1
        y3 = n-1
        int1 = integrate(ii, x0, y0, x1, y1)
        int2 = integrate(ii, x2, y2, x3, y3)

        dif.append(np.abs(int1 - int2))

    row_split = m / 2

    for i in range(2):
        x0 = 0
        y0 = i*col_split
        x1 = row_split
        y1 = (i+1)*col_split-1

        x2 = row_split
        y2 = y0
        x3 = m-1
        y3 = y1
        int1 = integrate(ii, x0, y0, x1, y1)
        int2 = integrate(ii, x2, y2, x3, y3)

        dif.append(np.abs(int1 - int2))

    if verbose:
        sys.stdout.write('\nDifferentiate sections::\n')
        sys.stdout.write(dif.__str__())

    return dif


def hist_hough(img, n_bins, verbose=False):
    """
    Computes the histogram of hough line magnitudes.

    :param img: Image
    :type img: nparray

    :param n_bins: Number of bins
    :type n_bins: int

    :return: histogram of hough lines magnitudes
    """
    # Find the hough lines
    edges = canny(img, 2, 1, 25)
    if verbose:
        sys.stdout.write('\nCanny Image:\n')
        sys.stdout.write(edges.__str__())

    theta = np.array([i*pi/100 for i in range(100)])
    lines = probabilistic_hough_line(edges, threshold=10, line_length=8, line_gap=3, theta=theta)

    # Compute the lines magnitudes
    magnitudes = []
    for l in lines:
        magnitudes.append(np.linalg.norm(l))

    if verbose:
        sys.stdout.write('\nMagnitudes:\n')
        sys.stdout.write(magnitudes.__str__())

    # Compute a histogram of line magnitudes
    hist, _ = np.histogram(magnitudes, bins=n_bins)

    if verbose:
        sys.stdout.write('\nHistogram of Hough:\n')
        sys.stdout.write(hist.__str__())

    return hist, edges