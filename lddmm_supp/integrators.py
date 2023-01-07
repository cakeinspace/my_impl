

from jax import lax



def odeint_rk4(f, y0, t, *args):
    """
    rk4 integration scheme
    """
    def step(state, t):
        y_prev, t_prev = state
        h = t - t_prev
        k1 = h * f(y_prev, t_prev,  *args)
        k2 = h * f(y_prev + k1/2., t_prev + h/2.,  *args)
        k3 = h * f(y_prev + k2/2., t_prev + h/2.,  *args)
        k4 = h * f(y_prev + k3, t + h,  *args)
        y = y_prev + 1./6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return (y, t), y
    _, ys = lax.scan(step, (y0, t[0]), t[1:])
    return ys


def odeint_euler(f, y0, t,  *args):
    """
    euler scheme
    """
    def step(state, t):
        y_prev, t_prev = state
        dt = t - t_prev
        y = y_prev + dt * f(y_prev, t_prev,  *args)
        return (y, t), y
    _, ys = lax.scan(step, (y0, t[0]), t[1:])
    return ys

def odeint_rk4_phiinv_naive(f, phi_id, t, v_int):
    """
    rk4 integration scheme
    """
    phi_f = phi_id
    for i in range(1, len(t)):
        h = t[i] - t[i-1]
        k1 = h * f(phi_f, t[i-1],  v_int[i-1])
        k2 = h * f(phi_f + k1/2.,  t[i-1] + h/2.,  v_int[i-1])
        k3 = h * f(phi_f + k2/2.,  t[i-1] + h/2.,  v_int[i-1])
        k4 = h * f(phi_f + k3, t[i] + h,  v_int[i-1])
        update =  1./6 * (k1 + 2 * k2 + 2 * k3 + k4)
        
        phi_f = phi_f + update
        
    return phi_f
