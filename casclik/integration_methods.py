"""Integration

This module contains functions to simplify integrations methods.

TODO:

"""
import casadi as cs


def get_euler_function(x, dx_function, dt):
    """Get a casadi function for the euler integration method."""
    return cs.Function("feuler", [x], [x+dx_function(x)*dt])


def get_rk4_function(x, dx_function, dt):
    """Get a casadi function for the RK4 integration method."""
    k1 = dx_function(x)
    k2 = dx_function(x + (dt/2.0)*k1)
    k3 = dx_function(x + (dt/2.0)*k2)
    k4 = dx_function(x + dt*k3)
    x_end = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return cs.Function("frk4", [x], [x_end])
