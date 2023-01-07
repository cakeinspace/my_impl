import jax.numpy as jnp
from jax import lax
from finite_differences import spatial_gradient 

def get_dm_dt_mermaid_formulation(m, t, K, smooth_across_components):
    
    """
    returns the update equations currently follows the mermaid evolution equations 
    an alternative is the equation given by mumford which are both the same 
    """
    
    v = smooth_across_components(m, K)

    m_x, m_y = m
    v_x, v_y = v

    
    dx_mxvx, _  = spatial_gradient(m_x*v_x)
    _, dy_mxvy = spatial_gradient(m_x*v_y)
    
    dx_vx, dy_vx = spatial_gradient(v_x)
    dx_vy, dy_vy = spatial_gradient(v_y)
    
    
    dx_myvx , _ = spatial_gradient(m_y*v_x)
    _, dy_myvy = spatial_gradient(m_y*v_y)
    
    
    dmx_dt = -dx_mxvx - dy_mxvy - dx_vx*m_x - dx_vy*m_x
    dmy_dt = -dx_myvx - dy_myvy - m_x*dy_vx - m_y*dy_vy

    
    dm_dt = jnp.asarray([dmx_dt, dmy_dt])
    
    return dm_dt

    
def set_derivatives_bc(m, bc_val):
    m = m.at[:, 0].set(bc_val)
    m = m.at[:, -1].set(bc_val)
    m = m.at[0, :].set(bc_val)
    m = m.at[-1, :].set(bc_val)
    return m

def get_dm_dt(m, t, K, smooth_across_components):
    """
    the equation is taken from mumford et al paper on epdiff equation
    where its given in a coordinates instead of vector notation 
    """
    
    v = smooth_across_components(m, K)

    m_x, m_y = m
    v_x, v_y = v

    dx_mx, dy_mx = spatial_gradient(m_x)
    dx_my, dy_my = spatial_gradient(m_y)
    
    dx_vx, dy_vx = spatial_gradient(v_x)
    dx_vy, dy_vy = spatial_gradient(v_y)
    
    dmx_dt = -(v_x*dx_mx + v_y*dy_mx + dx_vx*m_x + dy_vy*m_x + m_x*dx_vx + m_y*dx_vy)
    dmy_dt = -(v_x*dx_my + v_y*dy_my + dx_vx*m_y + dy_vy*m_y + m_x*dy_vx + m_y*dy_vy)
    
    dm_dt = jnp.asarray([dmx_dt, dmy_dt])
    
    return dm_dt


def get_dphi_dt(phi, t, v):
    """
    returns the update equation for the evolution of the phiinv mapping 
    """
    
    phix, phiy = phi
    v_x, v_y = v
    
    dx_phix, dy_phix = spatial_gradient(phix)
    dx_phiy, dy_phiy = spatial_gradient(phiy)
        
    dphix_dt = dx_phix*v_x + dy_phix*v_y
    dphiy_dt = dx_phiy*v_x + dy_phiy*v_y
    
    dphi_dt = -jnp.asarray([dphix_dt, dphiy_dt])
    
    return dphi_dt


def advect_map_forward_svf_rk4(f, phi0, t, v_0):
    
    """
    advects the phiinv using the update equation given above 
    """
    def step(state, t):

        y_prev, t_prev = state
        h = t - t_prev
        k1 = h * f(y_prev, t_prev,  v_0)
        k2 = h * f(y_prev + k1/2., t_prev + h/2.,  v_0)
        k3 = h * f(y_prev + k2/2., t_prev + h/2.,  v_0)
        k4 = h * f(y_prev + k3, t + h,  v_0)
        update = 1./6 * (k1 + 2 * k2 + 2 * k3 + k4)
        
        y = y_prev + update 
        return (y, t), y
    
    
    (phi_f, _), _ = lax.scan( step, (phi0, t[0]),  t[1:])
    return phi_f



def advect_map_forward(f, phi0, t, v_int):
    
    """
    advects the phiinv using the update equation given above 
    """
    def step(state, ti):
        i, t_curr = ti
        y_prev, t_prev, v_prev = state
        h = t[i] - t_prev
        k1 = h * f(y_prev, t_prev,  v_int[i])
        k2 = h * f(y_prev + k1/2., t_prev + h/2.,  v_int[i])
        k3 = h * f(y_prev + k2/2., t_prev + h/2.,  v_int[i])
        k4 = h * f(y_prev + k3, t[i] + h,  v_int[i])
        update = 1./6 * (k1 + 2 * k2 + 2 * k3 + k4)
        
        y = y_prev + update 
        return (y, t[i], v_int[i]), (t, v_int)
    
    #_, ys = lax.scan(step, (phi0, t[0], v_int[0]), (t[1:], v_int[1:]))
    init_val = ((phi0, t[0], v_int[0]), t, v_int)
    (phi_f, _, _), _ = lax.scan( step, (phi0, t[0], v_int[0]),  (jnp.arange(len(t)), t))
    return phi_f

