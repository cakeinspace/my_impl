

import jax.numpy as jnp
from jax import vmap
from finite_differences import spatial_gradient
import jax


def kncc(I, J, kernel):
    mode = "valid"
    Im = jax.scipy.signal.correlate(I, kernel, mode=mode)
    Jm = jax.scipy.signal.correlate(J, kernel, mode=mode)
    II = jax.scipy.signal.correlate(I * I, kernel, mode=mode) - Im ** 2
    JJ = jax.scipy.signal.correlate(J * J, kernel, mode=mode) - Jm ** 2
    IJ = jax.scipy.signal.correlate(I * J, kernel, mode=mode) - Im * Jm
    
    cost = -(IJ ** 2) / (II * JJ)
    cond_II = II < 1e-5
    cond_JJ = JJ < 1e-5
    cond = cond_II | cond_JJ
    
    cost = jnp.where( cond, 0., cost)
    #cost = 1-cost
    
    return jnp.mean(cost)

def ssd(I, J):
    """
    data matching term 
    """
    return jnp.mean((I-J)**2)

def _get_patches(im, window):
    w, h = im.shape
    idx_x = jnp.arange(0, w-window+1)
    idx_y = jnp.arange(0, h-window+1)
    grid = jnp.meshgrid(idx_x, idx_y)
    grid_idx = jnp.vstack((grid[0].flatten(), grid[1].flatten())).T
    return vmap(jax.lax.dynamic_slice , (None, 0, None))(im, grid_idx, [window, window])

def _compute_lncc(p1, p2, eps = 1e-20):
    
    p1m = p1 - p1.mean()
    p2m = p2 - p2.mean()
    num = (p1m*p2m).sum()
    denom = ((p1m**2).sum() * (p2m**2).sum())**.5 + eps
    return num/denom

def _compute_lncc_patches(p1_arr, p2_arr, eps = 1e-20):
    ns = p1_arr.shape[0]
    vals = vmap(_compute_lncc, (0, 0, None))(p1_arr, p2_arr, eps)
    return (1/ns)*(vals.sum())

def compute_lncc_filter(fix, mov, window, filter_zero):
    fix_patches = _get_patches(fix, window)
    mov_patches = _get_patches(mov, window)
    
    if filter_zero :
        fix_patches_mean = jnp.mean(fix_patches, axis = (1, 2))
        mov_patches_mean = jnp.mean(mov_patches, axis = (1, 2))

        idx_filter = (fix_patches_mean > 0.) & (mov_patches_mean > 0.)
        fix_patches_c = fix_patches[idx_filter]
        mov_patches_c = mov_patches[idx_filter]
    else:
        fix_patches_c = fix_patches
        mov_patches_c = mov_patches
    return _compute_lncc_patches(fix_patches_c, mov_patches_c)


def compute_lncc(fix, mov, window):
    fix_patches = _get_patches(fix, window)
    mov_patches = _get_patches(mov, window)
    

    return _compute_lncc_patches(fix_patches, mov_patches)



def compute_cc(fix, mov):
    fixm = fix - fix.mean()
    movm = mov - mov.mean()
    
    return 1-jnp.corrcoef(fixm.flatten(), movm.flatten())[0, 1]


def _get_lcc_loss_kernel(window):
    sigma = jnp.asarray([window, window])
    kernel_size = 2*sigma+1
    kernel = jnp.ones(kernel_size)/(jnp.product(kernel_size)**2)
    kernel = jnp.expand_dims(kernel, axis = (0, 1))
    return kernel

def _lcc_loss_2d(fix, mov, window ):
    kernel = _get_lcc_loss_kernel(window)
    padding = "VALID"
    
    fix_exp = jnp.expand_dims ( fix, axis = (0, 1))
    mov_exp = jnp.expand_dims ( mov, axis = (0, 1))
    
    mean_fix_exp = jax.lax.conv_general_dilated(fix_exp, kernel, window_strides = [1, 1], padding =padding)
    mean_mov_exp = jax.lax.conv_general_dilated(mov_exp, kernel, window_strides = [1, 1], padding =padding)
    variance_mov_exp = jax.lax.conv_general_dilated(mov_exp**2, kernel, window_strides = [1, 1], padding =padding) - mean_mov_exp**2
    variance_fix_exp = jax.lax.conv_general_dilated(fix_exp**2, kernel, window_strides = [1, 1], padding =padding) - mean_fix_exp**2
    
    mean_fix_mov_exp = jax.lax.conv_general_dilated(fix_exp*mov_exp, kernel, window_strides = [1, 1], padding =padding)
    cc = (mean_fix_mov_exp - mean_mov_exp * mean_fix_exp)**2 / (variance_mov_exp*variance_fix_exp + 1e-10)
    
    cc_mean = jnp.mean(cc)
    return -cc_mean


def _lcc_loss_2d_kernels(fix, mov, kernel):
    
    padding = "VALID"
    
    fix_exp = jnp.expand_dims ( fix, axis = (0, 1))
    mov_exp = jnp.expand_dims ( mov, axis = (0, 1))
    
    mean_fix_exp = jax.lax.conv_general_dilated(fix_exp, kernel, window_strides = [1, 1], padding =padding)
    mean_mov_exp = jax.lax.conv_general_dilated(mov_exp, kernel, window_strides = [1, 1], padding =padding)
    variance_mov_exp = jax.lax.conv_general_dilated(mov_exp**2, kernel, window_strides = [1, 1], padding =padding) - mean_mov_exp**2
    variance_fix_exp = jax.lax.conv_general_dilated(fix_exp**2, kernel, window_strides = [1, 1], padding =padding) - mean_fix_exp**2
    
    mean_fix_mov_exp = jax.lax.conv_general_dilated(fix_exp*mov_exp, kernel, window_strides = [1, 1], padding =padding)
    cc = (mean_fix_mov_exp - mean_mov_exp * mean_fix_exp)**2 / (variance_mov_exp*variance_fix_exp + 1e-10)
    cc_mean = jnp.mean(cc)
    return -cc_mean

def _lcc_loss_2d_exp(fix, mov, window ):
    kernel = _get_lcc_loss_kernel(window)
    padding = "VALID"
    
    fix_exp = jnp.expand_dims ( fix, axis = (0, 1))
    mov_exp = jnp.expand_dims ( mov, axis = (0, 1))
    
    mean_fix_exp = jax.lax.conv_general_dilated(fix_exp, kernel, window_strides = [1, 1], padding =padding)
    mean_mov_exp = jax.lax.conv_general_dilated(mov_exp, kernel, window_strides = [1, 1], padding =padding)
    variance_mov_exp = jax.lax.conv_general_dilated(mov_exp**2, kernel, window_strides = [1, 1], padding =padding) - mean_mov_exp**2
    variance_fix_exp = jax.lax.conv_general_dilated(fix_exp**2, kernel, window_strides = [1, 1], padding =padding) - mean_fix_exp**2
    
    mean_fix_mov_exp = jax.lax.conv_general_dilated(fix_exp*mov_exp, kernel, window_strides = [1, 1], padding =padding)
    cc = (mean_fix_mov_exp - mean_mov_exp * mean_fix_exp)**2 / (variance_mov_exp*variance_fix_exp + 1e-10)
    cc_mean = jnp.mean(cc)
    return 1-cc_mean, cc

def mk_lcc_loss(fix, mov, windows, weights):
    weights= jnp.asarray(weights)
    kernels = [_get_lcc_loss_kernel(w) for w in windows]
    lcc_loss = weights * jnp.asarray([jax.jit(_lcc_loss_2d_kernels)(fix, mov, k) for k in kernels])
    return jnp.mean(lcc_loss)


def mk_lcc_loss_kernels(fix, mov, kernels, weights):
    lcc_loss = 0
    for  k, w in zip(kernels, weights):
        lcc_loss += w * jax.jit(_lcc_loss_2d_kernels)(fix, mov, k)
    return jnp.mean(lcc_loss)

def ngf_loss_2d(fix, mov):
    epsilon = 1e-10
    dx_fixed, dy_fixed = spatial_gradient(fix)
    norm_fixed = (dx_fixed**2 + dy_fixed**2 + epsilon**2)**.5
    ng_fixed_image = jnp.asarray([dx_fixed, dy_fixed]) / norm_fixed
    dx_moving, dy_moving = spatial_gradient(mov)
    norm_moving = (dx_moving**2 + dy_moving**2 + epsilon**2)**.5
    ng_moving_image = jnp.asarray([dx_moving, dy_moving]) / norm_moving
    
    
    
    gf = -0.5*((ng_moving_image*ng_fixed_image).sum(0))**2
    return gf.mean()