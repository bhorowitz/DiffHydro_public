from functools import partial

# All steps take: L (rhs), u, dt, params -> u_new

def rk2_step(L, u, dt, params):
    k1 = L(u, params)
    u1 = u + 0.5 * dt * k1
    k2 = L(u1, params)
    return u + dt * k2

def ssprk3_step(L, u, dt, params):
    u1 = u + dt * L(u, params)
    u2 = 0.75 * u + 0.25 * (u1 + dt * L(u1, params))
    u3 = (1.0/3.0) * u + (2.0/3.0) * (u2 + dt * L(u2, params))
    return u3

def rk4_step(L, u, dt, params):
    k1 = L(u, params)
    k2 = L(u + 0.5*dt*k1, params)
    k3 = L(u + 0.5*dt*k2, params)
    k4 = L(u + dt*k3, params)
    return u + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

INTEGRATOR_DICT = {
    "RK2": rk2_step,
    "SSPRK3": ssprk3_step,
    "RK4": rk4_step,
}