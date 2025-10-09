import jax.numpy as jnp

#@jax.jit

#routines for sedov-taylor blast wave
def sedov(E0,rho,conf):
        #doesn't really work yet for non-square boxes...
        dV = (conf.box_size[0]/conf.mesh_shape[0])**3
        hc = int(conf.mesh_shape[0]/2)-1
        sol = jnp.zeros((5,conf.mesh_shape[0],conf.mesh_shape[1],conf.mesh_shape[2]))
        sol = sol.at[0].set(rho)
        sol = sol.at[-1].set(1.0E-3)
        sol = sol.at[4,hc,hc,hc].set(E0/dV)
        rmax = 3.0*(conf.box_size[0]/2.0)/4.0
        tf = jnp.sqrt((rmax/1.15)**5/E0)
        return sol, tf

def sedov_2d(E0,rho,conf):
        #doesn't really work yet for non-square boxes...
        dV = (conf.box_size[0]/conf.mesh_shape[0])**3
        hc = int(conf.mesh_shape[0]/2)-1
        sol = jnp.zeros((5,conf.mesh_shape[0],conf.mesh_shape[1]))
        sol = sol.at[0].set(rho)
        sol = sol.at[-1].set(1.0E-3)
        sol = sol.at[4,hc,hc].set(E0/dV)
        rmax = 3.0*(conf.box_size[0]/2.0)/4.0
        tf = jnp.sqrt((rmax/1.15)**5/E0)
        return sol, tf
#routines for polytropic profile

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def lane_emden_rhs(xi, y, n):
    theta, dtheta_dxi = y
    if theta > 0:
        d2theta_dxi2 = - (2/xi) * dtheta_dxi - theta**n
    else:
        d2theta_dxi2 = 0  # Stop integrating when theta becomes negative
    return [dtheta_dxi, d2theta_dxi2]

def solve_lane_emden(n, xi_max=10, num_points=1000):
    xi_vals = np.linspace(1e-6, xi_max, num_points)  # Avoid division by zero
    y0 = [1, 0]  # Initial conditions: theta(0) = 1, dtheta/dxi(0) = 0
    sol = solve_ivp(lane_emden_rhs, [xi_vals[0], xi_vals[-1]], y0, t_eval=xi_vals, args=(n,), method='RK45')
    return sol.t, sol.y[0]

def plot_density_profile(n):
    xi, theta = solve_lane_emden(n)
    plt.figure(figsize=(8, 5))
    plt.plot(xi, theta, label=f'Polytropic Index n={n}')
    plt.xlabel("Dimensionless Radius ξ")
    plt.ylabel("Dimensionless Density θ")
    plt.title("Density Profile of a Polytropic Sphere")
    plt.legend()
    plt.grid()
    plt.show()

def polytropic_density_3d(n, grid_size=256,scale_radius=1):
    #n is polytropic index
    xi, theta = solve_lane_emden(n)
    max_radius = xi[np.argmax(theta < 0)]  # Find the radius where density first becomes negative
    
    x = np.linspace(-max_radius, max_radius, grid_size)
    y = np.linspace(-max_radius, max_radius, grid_size)
    z = np.linspace(-max_radius, max_radius, grid_size)
    X, Y, Z = np.meshgrid(x, y, z)
    R = np.sqrt(X**2 + Y**2 + Z**2)*scale_radius
    
    # Interpolate theta values onto the 3D grid
    density = np.interp(R, xi, theta, left=0, right=0)

    return density