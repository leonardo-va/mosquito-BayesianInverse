import numpy as np
import sunode
from sunode.wrappers.as_pytensor import solve_ivp
import pytensor.tensor as pt
import pymc as pm
import arviz as az
from matplotlib import pyplot as plt

times = np.linspace(0,100,30)
lynx_data = np.array([
    4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3, 9.1, 7.4,
    8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6
])
hare_data = np.array([
    30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4,
    27.1, 40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7
])
lynx_data = times
hare_data = times
params_standalone = { 
    'alpha': (),
    'lf_M': ()
}


states = {
    'E': (),
    'J': (),
    'S_M': (),
    'E_M': (),
    'I_M': (),
    'S_H': (),
    'E_H': (),
    'I_H': (),
    'R_H': ()
}

def lotka_volterra(t, y, p):
    """Right hand side of Lotka-Volterra equation.

    All inputs are dataclasses of sympy variables, or in the case
    of non-scalar variables numpy arrays of sympy variables.
    """
    return {
        'E': 3*(y.S_M+y.E_M+y.I_M) - 0.6 * y.E - (0.6/0.875 - 0.6) * y.E,
        'J': 0.6 * y.E - 0.09 * y.J -p.alpha*y.J**2 - (0.09/0.75 - 0.09) * y.J,
        'S_M': 0.5*0.09 * y.J - 0.2*0.9*y.S_M*y.I_H/(y.S_M+y.E_H+y.I_H+y.R_H) - 1/(p.lf_M*150)*y.S_M,
        'E_M': 0.2*0.9*y.S_M*y.I_H/(y.S_M+y.E_H+y.I_H+y.R_H) - 1/30*y.E_M - 1/(p.lf_M*150)*y.E_M,
        'I_M': 1/30*y.E_M - 1/(p.lf_M*150)*y.I_M,
        'S_H': 12 - 0.2*0.8*y.S_M*y.I_M/(y.S_M+y.E_M+y.I_M) - 0.001*y.S_M,
        'E_H': 0.2*0.8*y.S_M*y.I_M/(y.S_M+y.E_M+y.I_M) - 0.4*y.E_H - 0.001*y.E_H,
        'I_H': 0.4*y.E_H - 0.1818*y.I_H - 0.001*y.I_H,
        'R_H': 0.1818*y.I_H - 0.001 * y.R_H
    }
problem = sunode.SympyProblem(
    params=params_standalone,
    states=states,
    rhs_sympy=lotka_volterra,
    derivative_params=()
)

solver = sunode.solver.Solver(problem, solver='BDF')

y0_standalone = np.zeros((), dtype=problem.state_dtype)
y0_standalone['E'] = 10000
y0_standalone['J'] = 10000
y0_standalone['S_M'] = 50000
y0_standalone['E_M'] = 0
y0_standalone['I_M'] = 10000
y0_standalone['S_H'] = 10000
y0_standalone['E_H'] = 0
y0_standalone['I_H'] = 0
y0_standalone['R_H'] = 0

solver.set_params_dict({
    'alpha': 0.000002,
    'lf_M': 0.067
})

output = solver.make_output_buffers(times)
solver.solve(t0=times[0], tvals=times, y0=y0_standalone, y_out=output)

gaussian_noise = np.random.normal(0,2000,output.shape)
measured = output + gaussian_noise
lynx_data = measured[:,1]
hare_data = measured[:,2]
plt.plot(times, output[:,1])
plt.show()

with pm.Model() as model:
    # hares_start = pm.HalfNormal('hares_start', sigma=50)
    # lynx_start = pm.HalfNormal('lynx_start', sigma=50)
  
    e_start = 10000
    j_start = 10000
    sm_start = 50000
    em_start = 0
    im_start = 10000
    sh_start = 10000
    eh_start = 0
    ih_start = 0
    rh_start = 0

    # ratio = pm.Beta('ratio', alpha=0.5, beta=0.5)

    # fixed_hares = pm.HalfNormal('fixed_hares', sigma=50)
    # fixed_lynx = pm.Deterministic('fixed_lynx', ratio * fixed_hares)

    # period = pm.Gamma('period', mu=10, sigma=1)
    # freq = pm.Deterministic('freq', 2 * np.pi / period)

    # log_speed_ratio = pm.Normal('log_speed_ratio', mu=0, sigma=0.1)
    # speed_ratio = np.exp(log_speed_ratio)

    # # Compute the parameters of the ode based on our prior parameters
    # alpha = pm.Deterministic('alpha', freq * speed_ratio * ratio)
    # beta = pm.Deterministic('beta', freq * speed_ratio / fixed_hares)
    # gamma = pm.Deterministic('gamma', freq / speed_ratio / ratio)
    # delta = pm.Deterministic('delta', freq / speed_ratio / fixed_hares / ratio)
    alpha = pm.HalfNormal('alpha', 1)
    lf_M = pm.HalfNormal('lf_M', 1)

with model:
    y0 = {
        # The initial number of hares is the random variable `hares_start`,
        # and it has shape (), so it is a scalar value.
        'E': (e_start, ()),
        'J': (j_start, ()),
        'S_M': (sm_start, ()),
        'E_M': (em_start, ()),
        'I_M': (im_start, ()),
        'S_H': (sh_start, ()),
        'E_H': (eh_start, ()),
        'I_H': (ih_start, ()),
        'R_H': (rh_start, ()),
    }

    params = {
        'alpha': (alpha, ()),
        'lf_M': (lf_M, ()),
        # Parameters (or initial states) do not have to be random variables,
        # they can also be fixed numpy values. In this case the shape
        # is infered automatically. Sunode will not compute derivatives
        # with respect to fixed parameters or initial states.
        'unused_extra': np.zeros(5),
    }

with model:
    # from sunode.wrappers.as_pytensor import solve_ivp
        solution, *_ = solve_ivp(
        y0=y0,
        params=params,
        rhs=lotka_volterra,
        # The time points where we want to access the solution
        tvals=times,
        t0=times[0],
    )
    
with model:
    # We can access the individual variables of the solution using the
    # variable names.
    # pm.Deterministic('hares_mu', solution['hares'])
    # pm.Deterministic('lynxes_mu', solution['lynxes'])

    # sd = pm.HalfNormal('sd')
    pm.Normal('juv', mu=solution['J'], sigma=2000, observed=hare_data)
    pm.Normal('totalm', mu=solution['S_M'], sigma=2000, observed=lynx_data)

with model:
    trace = pm.sample()