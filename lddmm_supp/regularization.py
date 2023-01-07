import jax.numpy as jnp
from jax import vmap
from functools import partial
from grid_utils import identity_mapping

def get_normalized_gaussian_kernel(X, mu, sig):
    """
    returns the normalized gaussian kernel for smoothing the vector momentum to get the 
    velocity field 
    """
    
    g = jnp.exp(-jnp.power(X[0,:,:]-mu[0],2.)/(2*jnp.power(sig[0],2.))
                   - jnp.power(X[1,:, :] - mu[1], 2.) / (2 * jnp.power(sig[1], 2.)))
    g = g/g.sum()
    return g





def smooth(v, K,  pad):
   
    ### add padding here ###
    pad = 2*pad
    required_padding = [pad, pad]
    
    v = jnp.pad(v, required_padding, mode="edge")
    K = jnp.pad(K, required_padding, mode = "edge")
    
    freq = jnp.fft.fftn(jnp.fft.ifftshift(v))
    freq_kernel = jnp.fft.fftn(K)
    convolved = freq*freq_kernel
    v_smoothed = jnp.fft.ifftn(convolved).real
    return v_smoothed[pad:-pad, pad:-pad]


def get_smoother(smooth , flow_sigma):
    smooth_across_components = vmap(partial(smooth, pad=flow_sigma), (0, None)) ## apply the smoother across the velocity field components, v is of shape (2, w, h) and this applies it across the component
    smooth_across_time_and_components = vmap(vmap(partial(smooth, pad=flow_sigma), (0, None)), (0, None)) ## apply smoother across the velocity field time and space components t, 2, w, h
    return smooth_across_components, smooth_across_time_and_components 


def smooth_dum_dum(v, K):
    return jax.scipy.signal.convolve(v, K, mode="same")


def get_smoother_dum_dum(smooth):
    smooth_across_components = vmap(partial(smooth), (0, None)) ## apply the smoother across the velocity field components, v is of shape (2, w, h) and this applies it across the component
    smooth_across_time_and_components = vmap(vmap(partial(smooth), (0, None)), (0, None)) ## apply smoother across the velocity field time and space components t, 2, w, h
    return smooth_across_components, smooth_across_time_and_components 



## create all the padding functions before hand and then use them later 

def _smooth_adaptive_single_kernel(v, K_s, w_s, pad_s):
    
    """
    i can create and pass padding functions before hand 
    """
   
    wsqrt_v = jnp.sqrt(w_s)*v
    
    pad_s = 2*pad_s
    required_padding = [pad_s, pad_s]
    
    wsqrt_v = jnp.pad(wsqrt_v, required_padding, mode="edge")
    K_s = jnp.pad(K_s, required_padding, mode = "edge")
    freq = jnp.fft.fftn(jnp.fft.ifftshift(wsqrt_v))
    freq_kernel = jnp.fft.fftn(K_s)
    convolved = freq*freq_kernel
    v_smoothed = jnp.fft.ifftn(convolved).real
    return jnp.sqrt(w_s)*v_smoothed[pad_s:-pad_s, pad_s:-pad_s]
    
def get_adaptive_smoothing_funcs(sigma_array):
    n_sigmas = len(sigma_array)
    smoothing_funcs = []
    for i in range(n_sigmas):
        sigma = sigma_array[i]
        smoother =  partial(_smooth_adaptive_single_kernel, pad_s = sigma)
        smoothing_funcs.append(smoother)
    
    return smoothing_funcs

def adaptive_smoothing(v, K, weights, smoothing_funcs):
    v_sm = []
    for i in range(len(smoothing_funcs)):
        
        v_sm.append(smoothing_funcs[i](v, K[i, :, :], weights[:, :, i]))
    v_sm = jnp.asarray(v_sm)
    return v_sm.sum(0)




def get_smoothers_and_kernels(flow_sigma, w, h):
    smooth_across_components, smooth_across_time_and_components = get_smoother(smooth,  flow_sigma)
    id_map = identity_mapping((w, h))
    K = get_normalized_gaussian_kernel(id_map,(w//2, h//2), (flow_sigma, flow_sigma) )
    return smooth_across_components, smooth_across_time_and_components, K
