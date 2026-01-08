import numpy as np
import sunode
from sunode.wrappers.as_pytensor import solve_ivp
import pytensor.tensor as pt
import pymc as pm
import arviz as az
from matplotlib import pyplot as plt
import pickle
import sympy as sp
import temperature_dependent_parameters as tdp

times = np.linspace(0,100,30)

gt_parameters = {
    "delta_E": 0.6,
    "p_E": 0.875,
    "beta": 3,
    "alpha": 2e-06,
    "delta_J": 0.09,
    "p_J": 0.75,
    "omega": 0.5,
    "lf_M": 0.067,
    "a": 0.2,
    "b_M": 0.9,
    "EIP_M": 30,
    "Lambda": 12,
    "b_H": 0.8,
    "mu_H": 0.001,
    "alpha_H": 0.4,
    "gamma_H": 0.1818
}

params_standalone = { 
    "delta_E": (),
    "p_E": (),
    "beta": (),
    "alpha": (),
    "delta_J": (),
    "p_J": (),
    "omega": (),
    "lf_M": (),
    "a": (),
    "b_M": (),
    "EIP_M": (),
    "Lambda": (),
    "b_H": (),
    "mu_H": (),
    "alpha_H": (),
    "gamma_H": ()
}

def temperature_profile(time):
    time_15_0 = sp.asin(1/2) * 100 / sp.pi
    time_15_1 = (sp.pi - sp.asin(1/2)) * 100 / sp.pi
    result = sp.Piecewise(
        (15, (time < 0) | (time > 100)),  # Return 15 if time is out of bounds
        (30 * sp.sin((time_15_0 + time / 100 * (time_15_1 - time_15_0)) * sp.pi / 100), True)  # Otherwise
    )
    return result

def const_t_dep_param(t, funcparam):
    return funcparam

def lotka_volterra(t, y, p):
    """Right hand side of Lotka-Volterra equation.

    All inputs are dataclasses of sympy variables, or in the case
    of non-scalar variables numpy arrays of sympy variables.
    """
    return {
        'E': tdp.ConstantFunction().eval_sp(temperature_profile(t),p.beta)*(y.S_M+y.E_M+y.I_M) - p.delta_E * y.E - (p.delta_E/p.p_E - p.delta_E) * y.E,
        'J': p.delta_E * y.E - p.delta_J * y.J - p.alpha*y.J**2 - (p.delta_J/p.p_J - p.delta_J) * y.J,
        'S_M': p.omega*p.delta_J * y.J - p.a*p.b_M*y.S_M*y.I_H/(y.S_M+y.E_H+y.I_H+y.R_H) - 1/(p.lf_M*150)*y.S_M,
        'E_M': p.a*p.b_M*y.S_M*y.I_H/(y.S_M+y.E_H+y.I_H+y.R_H) - 1/p.EIP_M*y.E_M - 1/(p.lf_M*150)*y.E_M,
        'I_M': 1/p.EIP_M*y.E_M - 1/(p.lf_M*150)*y.I_M,
        'S_H': p.Lambda - p.a*p.b_H*y.S_M*y.I_M/(y.S_M+y.E_M+y.I_M) - p.mu_H*y.S_M,
        'E_H': p.a*p.b_H*y.S_M*y.I_M/(y.S_M+y.E_M+y.I_M) - p.alpha_H*y.E_H - p.mu_H*y.E_H,
        'I_H': p.alpha_H*y.E_H - p.gamma_H*y.I_H - p.mu_H*y.I_H,
        'R_H': p.gamma_H*y.I_H - p.mu_H * y.R_H
    }

y0 = {
    'E': (10000, ()),
    'J': (10000, ()),
    'S_M': (50000, ()),
    'E_M': (0, ()),
    'I_M': (10000, ()),
    'S_H': (10000, ()),
    'E_H': (0, ()),
    'I_H': (0, ()),
    'R_H': (0, ()),
}

noise_stddev = {
    'J': 10000,
    'S_M': 10000
}


states = {}
for state_name in y0:
       states[state_name] = ()

problem = sunode.SympyProblem(
    params=params_standalone,
    states=states,
    rhs_sympy=lotka_volterra,
    derivative_params=()
)

solver = sunode.solver.Solver(problem, solver='BDF')

y0_standalone = np.zeros((), dtype=problem.state_dtype)
for state_name in y0:
    y0_standalone[state_name] = y0[state_name][0]

solver.set_params_dict(gt_parameters)

output = solver.make_output_buffers(times)
solver.solve(t0=times[0], tvals=times, y0=y0_standalone, y_out=output)

gaussian_noise = np.random.normal(0,2000,output.shape)
measured = output + gaussian_noise
observed_quantities = {
    "J": output[:,1],
    "S_M": output[:,2]
}

measurements = {}
for observed_q in observed_quantities:
    noise = np.random.normal(0, noise_stddev[observed_q], observed_quantities[observed_q].shape)
    measurements[observed_q] = (observed_quantities[observed_q] + noise).copy()

plt.plot(times, output[:,1])
plt.show()
plt.plot(times, measurements["J"])

with pm.Model() as model:
   
    alpha = pm.HalfNormal('alpha', 1)
    lf_M = pm.HalfNormal('lf_M', 1)


    # params = {
    #     'alpha': (alpha, ()),
    #     'lf_M': (lf_M, ()),
    #     # Parameters (or initial states) do not have to be random variables,
    #     # they can also be fixed numpy values. In this case the shape
    #     # is infered automatically. Sunode will not compute derivatives
    #     # with respect to fixed parameters or initial states.
    #     'unused_extra': np.zeros(()),
    # }

    params = {
        "delta_E": 0.6,
        "p_E": 0.875,
        "beta": 3,
        "alpha": (alpha, ()),
        "delta_J": 0.09,
        "p_J": 0.75,
        "omega": 0.5,
        "lf_M": (lf_M, ()),
        "a": 0.2,
        "b_M": 0.9,
        "EIP_M": 30,
        "Lambda": 12,
        "b_H": 0.8,
        "mu_H": 0.001,
        "alpha_H": 0.4,
        "gamma_H": 0.1818
    }

    solution, *_ = solve_ivp(
    y0=y0,
    params=params,
    rhs=lotka_volterra,
    # The time points where we want to access the solution
    tvals=times,
    t0=times[0]
    )
    

    # We can access the individual variables of the solution using the
    # variable names.
    # pm.Deterministic('hares_mu', solution['hares'])
    # pm.Deterministic('lynxes_mu', solution['lynxes'])
    pm.Normal('juv', mu=solution['J'], sigma=noise_stddev['J'], observed=measurements['J'])
    pm.Normal('totalm', mu=solution['S_M'], sigma=noise_stddev['S_M'], observed=measurements['S_M'])

with model:
    trace = pm.sample()

with open('trace.pkl', 'wb') as pymc_result_file:
    pickle.dump(trace, pymc_result_file)
# -----------------------------
# # 4. ArviZ Visualizations
# # -----------------------------
# az.summary(trace, round_to=2)
# az.summary(trace, kind="stats")
# az.plot_trace(trace)
# # Posterior parameter distributions
# az.plot_posterior(trace, var_names=["alpha", "lf_M"])
# plt.show()
# # Posterior predictive checks (built-in)