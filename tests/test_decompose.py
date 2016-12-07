import os
import numpy as np
from scipy.misc import imread
#import cudasift as cs


def cart2polar(x, y):
    theta = np.arctan2(y, x)
    return theta

def index_coords(data, origin=None):
    """Creates x & y coords for the indicies in a numpy array "data".
    "origin" defaults to the center of the image. Specify origin=(0,0)
    to set the origin to the lower left corner of the image."""
    ny, nx = data.shape[:2]
    if origin is None:
        origin_x, origin_y = nx // 2, ny // 2
    else:
        origin_x, origin_y = origin
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x -= origin_x
    y -= origin_y
    return x, y

def reproject_image_into_polar(data, origin=None):
    """Reprojects a 3D numpy array ("data") into a polar coordinate system.
    "origin" is a tuple of (x0, y0) and defaults to the center of the image."""
    ny, nx = data.shape[:2]
    if origin is None:
        origin = (nx//2, ny//2)

    # Determine that the min and max r and theta coords will be...
    x, y = index_coords(data, origin=origin)
    theta = cart2polar(x, y)
    theta[theta < 0] += 2 * np.pi
    return theta


def coupled_decomposition(data, originx, originy, M=4, radial_size=720):
    """
    Apply coupled decomposition to two 2d images=.

    sdata : ndarray
            (n,m) array of values to decompose

    ddata : ndarray
            (j,k) array of values to decompose

    sorigin : tuple
              in the form (x,y)

    dorigin : tuple
              in the form (x,y)

    """

    # Create membership arrays for each input image
    membership = np.ones(data.shape)

    # Project the image into a polar coordinate system centered on p_{1}
    thetas = reproject_image_into_polar(data, origin=(int(originx),
                                                       int(originy)))

    h, w = data.shape
    # Compute the mean profiles for each radial slice
    means = np.empty(radial_size)

    # Pass to C
    cs.PyRadialMean(radial_size, h, w, data, thetas, means)
    print(means)

data = imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AS15-M-0295_SML.png'))
x, y = data.shape
originx = int(x / 2)
originy = int(y / 2)
coupled_decomposition(data, originx, originy)
