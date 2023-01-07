import jax.numpy as jnp
from jax.experimental import optimizers
from tqdm.notebook import tqdm
from grid_utils import identity_mapping
import jax
from jax import value_and_grad
import matplotlib.pyplot as plt
from plotting import show_warp_field

def optimize_gd_naive(func, args, step_size, num_epochs, debug = True, debug_plot = False):

    m0, fix, mov, t, sigma, K, integrator, momentum_update, phiinv_update, smooth_across_components, smooth_across_time_and_components, advect_phiinv, data_matching_measure, regularization = args
    
    data_matching_loss = []
    regularization_loss = []
    total_loss = []
    grads_norm = []
    
    for epoch in tqdm(range(num_epochs)):
        
        loss, grads = func(m0, K, fix, mov, t, 
                           sigma,  integrator, momentum_update, phiinv_update, smooth_across_components, smooth_across_time_and_components, 
                           advect_phiinv, data_matching_measure, regularization)
        
        
        m0 = m0 - step_size*grads 
        
        total_loss.append(loss)
        grads_norm.append(jnp.mean(step_size*grads))
        
        if debug:
            
            warped, v_int, phi_ad_x, phi_ad_y = get_results([m0]+args[1:], plot = debug_plot)
            dm = (1/sigma)**2 * data_matching(fix, warped, data_matching_measure)
            reg = 0.5 * regularization(m0, K, smooth_across_components)
            
            data_matching_loss.append(dm)
            regularization_loss.append(reg)
    diagnostics = [total_loss, data_matching_loss, regularization_loss, grads_norm]
    return m0, diagnostics



def get_results(args, plot = True):
    m0, fix, mov, t, sigma, K, integrator, momentum_update, phiinv_update, smooth_across_components,smooth_across_time_and_components, advect_phiinv, data_matching_measure, regularization = args
    w , h = fix.shape
    x0 = jnp.arange(0, w)
    x1 = jnp.arange(0, h)
    m_int = integrator(momentum_update, m0, t, K, smooth_across_components)
    v_int = smooth_across_time_and_components(m_int, K)
    phi_id = identity_mapping(fix.shape)
    phi_ad_x, phi_ad_y = advect_phiinv(phiinv_update, phi_id, t, v_int)

    coords = jnp.asarray([phi_ad_y, phi_ad_x])
    warped = jax.scipy.ndimage.map_coordinates(mov, coords, order= 1)

    if plot:

        u, v = m0
        f, a = plt.subplots(1, 6, figsize =(30, 6))
        a[0].imshow(fix)
        a[1].imshow(mov)
        a[3].quiver(x0, x1, u, v, headwidth = 15)
        a[3].invert_yaxis()
        a[2].imshow(fix, alpha = .2)
        a[2].imshow(mov, alpha = .4)
        a[4].imshow(warped)
        a[5].imshow(fix, alpha = .2)
        a[5].imshow(warped, alpha = .4)


        show_warp_field(jnp.asarray([phi_ad_y, phi_ad_x]), interval=1, shape = fix.shape, size = (4, 4),limit_axis=False, show_axis=True, plot_separately = False)
        plt.show()
    return warped, v_int, phi_ad_x, phi_ad_y

def optimize(func, opt_func, args, step_size, num_epochs, debug = True, debug_plot = False):

    m0, fix, mov, t, sigma, K, integrator, momentum_update, phiinv_update, smooth_across_components, smooth_across_time_and_components, advect_phiinv, data_matching_measure, regularization = args
    opt_init, opt_update, get_params = opt_func(step_size)
    opt_state = opt_init(m0)

    data_matching_loss = []
    regularization_loss = []
    total_loss = []
    grads_norm = []
    
    for epoch in tqdm(range(num_epochs)):
        
        params = get_params(opt_state)
        loss, grads = func(params, K, fix, mov, t, 
                           sigma,  integrator, momentum_update, phiinv_update, smooth_across_components, smooth_across_time_and_components, 
                           advect_phiinv, data_matching_measure, regularization)
        
        opt_state = opt_update(epoch, grads, opt_state)
        params = get_params(opt_state)
        
        total_loss.append(loss)
        grads_norm.append(jnp.mean(step_size*grads))
        
        if debug:
            warped, v_int, phi_ad_x, phi_ad_y = get_results([params]+args[1:], debug_plot)
            dm = (1/sigma)**2 * data_matching(fix, warped, data_matching_measure)
            reg = 0.5 * regularization(get_params(opt_state), K, smooth_across_components)
            
            data_matching_loss.append(dm)
            regularization_loss.append(reg)
        
    diagnostics = [total_loss, data_matching_loss, regularization_loss, grads_norm]
    return get_params(opt_state), diagnostics

def data_matching(I, J, sim_measure):
    """
    data matching term 
    """
    return sim_measure(I, J)



def regularization(m0, K, smooth_across_components):
    
    v0 = smooth_across_components(m0, K)
    v_total  = jnp.sum(v0.flatten()*v0.flatten())
    return v_total



def cost_functional(m0, K, fix, mov, t, sigma, integrator, momentum_update, phiinv_update, smooth_across_components,smooth_across_time_and_components, advect_phiinv, data_matching_measure, 
                   regularization):
    """
    the error term which is data matching + regularization 
    """
    
    def shoot(m0, t, K, integrator, momentum_update, phiinv_update, smooth_across_time_and_components, advect_phiinv):
    
        m_int = integrator(momentum_update, m0, t, K, smooth_across_components)
        v_int = smooth_across_time_and_components(m_int, K)
        
        phi_id = identity_mapping(fix.shape) 
        phi_ad_x, phi_ad_y = advect_phiinv(phiinv_update, phi_id, t, v_int)
        
        coords = jnp.asarray([phi_ad_y, phi_ad_x])
        warped = jax.scipy.ndimage.map_coordinates(mov, coords, order= 1)
        
        return warped

    warped = shoot(m0, t, K,  integrator, momentum_update, phiinv_update, smooth_across_time_and_components, advect_phiinv)
    
    dm = (1/sigma)**2 * data_matching(fix, warped, data_matching_measure)
    
    reg = 0.5 * regularization(m0, K, smooth_across_components)
    return  dm + reg


def cost_functional_svf(m0, K, fix, mov, t, sigma, integrator, momentum_update, phiinv_update, smooth_across_components, advect_phiinv, data_matching_measure, 
                   regularization):
    """
    the error term which is data matching + regularization 
    """
    
    def shoot(m0, t, K, integrator, momentum_update, phiinv_update, advect_phiinv):
    
        v_0 = smooth_across_components(m0, K)
        phi_id = identity_mapping(fix.shape) 
        phi_ad_x, phi_ad_y = advect_phiinv(phiinv_update, phi_id, t, v_0)
        coords = jnp.asarray([phi_ad_y, phi_ad_x])
        warped = jax.scipy.ndimage.map_coordinates(mov, coords, order= 1)
        
        return warped

    warped = shoot(m0, t, K,  integrator, momentum_update, phiinv_update, advect_phiinv)
    
    dm = (1/sigma)**2 * data_matching(fix, warped, data_matching_measure)
    
    reg = 0.5 * regularization(m0, K, smooth_across_components)
    return  dm + reg


