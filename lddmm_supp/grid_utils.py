

import math
from scipy.ndimage import gaussian_filter
import numpy as np
from scipy.ndimage import zoom
import jax.numpy as jnp
import plotting as plot
import grid_utils as gu



def identity_mapping(shape):
    """
    returns the identity map to initialize the phiinv 
    """
    
    x1 = np.asarray([float(i) for i in range(shape[0])])
    y1 = np.asarray([float(i)for i in range(shape[1])])
    return jnp.asarray(jnp.meshgrid(x1, y1, indexing = "xy"))
    
    

## utilities function 
def change_resolution(img, resolution, sigma, order=1, ndim = 2):
    """
    change image's resolution
    Parameters
    ----------
    resolution : int
        how much to magnify
        if resolution is 2, the shape of the image will be halved
    sigma : float
        standard deviation of gaussian filter for smoothing
    order : int
        order of interpolation
    Returns
    -------
    img : ScalarImage
        zoomed scalar image
    """
    if resolution != 1:
        blurred_data = gaussian_filter(img, sigma)
        ratio = [1 / float(resolution)] * ndim
        data = zoom(blurred_data, ratio, order=order)
    elif resolution == 1:
        data = gaussian_filter(img, sigma)
    return data

def change_scale(img, maximum_value):
    data = maximum_value * img / np.max(img)
    return data


def zoom_grid(grid, resolution,shape0, ndim = 2  ):
    shape = grid.shape[1:]
    if resolution != 1:
        interpolated_grid = np.zeros((ndim,) + shape0)
        for i in range(ndim):
            interpolated_grid[i] = interpolate_mapping(
                grid[i], np.array(shape0, dtype=np.int32)
            ) * (shape0[i] - 1) / (shape[i] - 1)
        return interpolated_grid
    else:
        return grid

def interpolate_mapping(func,  target_shape):
    return interpolate2d(func, func.shape[0], func.shape[1], target_shape)
    

def interpolate2d( func,  xlen_now,  ylen_now,  target_shape):
    xlen_target = target_shape[0]
    ylen_target = target_shape[1]

    interpolated = np.zeros((xlen_target, ylen_target))

    for x in range(xlen_target):
        xi = x * (xlen_now - 1) / (xlen_target - 1.)
        for y in range(ylen_target):
            yi = y * (ylen_now - 1) / (ylen_target - 1.)
            interpolated[x,y] = bilinear_interpolation(func, xi, yi, xlen_now, ylen_now)

    return interpolated


def bilinear_interpolation(func, x,  y,  xlen,  ylen):
    """
    Bilinear interpolation at a given position in the image.
    Parameters
    ----------
    func : double array
        Input function.
    x, y : double
        Position at which to interpolate.
    Returns
    -------
    value : double
        Interpolated value.
    """

    
    x0 = math.floor(x)
    x1 = math.ceil(x)
    y0 = math.floor(y)
    y1 = math.ceil(y)

    dx = x - x0
    dy = y - y0

    f0 = (1 - dy) * getValue2d(func, x0, y0, xlen, ylen, 'N') + dy * getValue2d(func, x0, y1, xlen, ylen, 'N')
    f1 = (1 - dy) * getValue2d(func, x1, y0, xlen, ylen, 'N') + dy * getValue2d(func, x1, y1, xlen, ylen, 'N')

    return (1 - dx) * f0 + dx * f1


def getValue2d(func,  x,  y,  xlen,  ylen,  mode='N'):
    if mode == 'N':
        if x < 0:
            x = 0
        elif x > xlen - 1:
            x = xlen - 1

        if y < 0:
            y = 0
        elif y > ylen - 1:
            y = ylen - 1
    elif mode == 'C':
        if x < 0 or x > xlen - 1 or y < 0 or y > ylen - 1:
            return 0
    
    return func[x , y]

        
def get_mapping_grid_final(velocity_list, shape0=(120, 120)):
    phi_id = jnp.asarray(gu.identity_mapping(shape0))
    phi = phi_id

    for v_int, resolution in zip(velocity_list,resolutions):
        vlist = []
        for vi in tqdm(v_int):
            v_zoomed = gu.zoom_grid(vi, resolution, ndim = 2, shape0 = shape0)
            vlist.append(v_zoomed)
        vlist = jnp.asarray(vlist)

        phi_ad_x, phi_ad_y = advect_map_forward(get_dphi_dt, phi, t, vlist)
        plot.show_warp_field(jnp.asarray([phi_ad_x, phi_ad_y]), interval=2, shape = fix.shape, size = (4, 4),limit_axis=False, show_axis=True, plot_separately = False)
        phi = jnp.asarray([phi_ad_x, phi_ad_y])
    return phi