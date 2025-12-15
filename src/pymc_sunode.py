import numpy as np
import sunode
import sunode.wrappers.as_pytensor
import pymc as pm
import arviz as az
from matplotlib import pyplot as plt

times = np.linspace(0,100,30)

params = {
    'alpha': (),
    'lf_M': ()
}

states = {
    "E": (),
    "J": (),
    "S_M": (),
    "E_M": (),
    "I_M": (),
    "S_H": (),
    "E_H": (),
    "I_H": (),
    "R_H": ()
    }

def mosquito_rhs(t, u, p):
    delta_E =0.6
    p_E= 0.875
    beta= 3
    # alpha= theta[0]
    delta_J= 0.09
    p_J= 0.75
    omega= 0.5
    # lf_M= theta[1]
    a= 0.2
    b_M = 0.9
    EIP_M= 30
    Lambda= 12
    b_H= 0.8
    mu_H= 0.001
    alpha_H= 0.4
    gamma_H= 0.1818
    return{
        "E": beta*(u[2]+u[3]+u[4]) - delta_E * u[0] - (delta_E/p_E - delta_E) * u[0],
        "J": delta_E * u[0] - delta_J * u[1] -p.alpha*u[1]**2 - (delta_J/p_J - delta_J) * u[1],
        "S_M": omega*delta_J * u[1] - a*b_M*u[2]*u[7]/(u[5]+u[6]+u[7]+u[8]) - 1/(p.lf_M*150)*u[2],
        "E_M": a*b_M*u[2]*u[7]/(u[5]+u[6]+u[7]+u[8]) - 1/EIP_M*u[3] - 1/(p.lf_M*150)*u[3],
        "I_M": 1/EIP_M*u[3] - 1/(p.lf_M*150)*u[4],
        "S_H": Lambda - a*b_H*u[5]*u[4]/(u[2]+u[3]+u[4]) - mu_H*u[5],
        "E_H": a*b_H*u[5]*u[4]/(u[2]+u[3]+u[4]) - alpha_H*u[6] - mu_H*u[6],
        "I_H": alpha_H*u[6] - gamma_H*u[7] - mu_H*u[7],
        "R_H": gamma_H*u[7] - mu_H * u[8]
    }

problem = sunode.SympyProblem(
    params=params,
    states=states,
    rhs_sympy=mosquito_rhs,
    derivative_params=()
)
solver = sunode.solver.Solver(problem, compute_sens=False, solver='BDF')

y0 = np.zeros((), dtype=problem.state_dtype)
y0['E'] = 10000
y0['J'] = 10000
y0['S_M'] = 50000
y0['E_M'] = 0
y0['I_M'] = 10000
y0['S_H'] = 10000
y0['E_H'] = 0
y0['I_H'] = 0
y0['R_H'] = 0

solver.set_params_dict({
    'alpha': 0.1,
    'beta': 0.2,
    'gamma': 0.3,
    'delta': 0.4,
})

output = solver.make_output_buffers(tvals)
solver.solve(t0=0, tvals=tvals, y0=y0, y_out=output)

with pm.Model() as model:
    hares_start = pm.HalfNormal('hares_start', sigma=50)
    lynx_start = pm.HalfNormal('lynx_start', sigma=50)
    
    ratio = pm.Beta('ratio', alpha=0.5, beta=0.5)
        
    fixed_hares = pm.HalfNormal('fixed_hares', sigma=50)
    fixed_lynx = pm.Deterministic('fixed_lynx', ratio * fixed_hares)
    
    period = pm.Gamma('period', mu=10, sigma=1)
    freq = pm.Deterministic('freq', 2 * np.pi / period)
    
    log_speed_ratio = pm.Normal('log_speed_ratio', mu=0, sigma=0.1)
    speed_ratio = np.exp(log_speed_ratio)
    
    # Compute the parameters of the ode based on our prior parameters
    alpha = pm.Deterministic('alpha', freq * speed_ratio * ratio)
    beta = pm.Deterministic('beta', freq * speed_ratio / fixed_hares)
    gamma = pm.Deterministic('gamma', freq / speed_ratio / ratio)
    delta = pm.Deterministic('delta', freq / speed_ratio / fixed_hares / ratio)
    
    y_hat, _, problem, solver, _, _ = sunode.wrappers.as_pytensor.solve_ivp(
        y0={
        # The initial conditions of the ode. Each variable
        # needs to specify a PyTensor or numpy variable and a shape.
        # This dict can be nested.
            'hares': (hares_start, ()),
            'lynx': (lynx_start, ()),
        },
        params={
        # Each parameter of the ode. sunode will only compute derivatives
        # with respect to PyTensor variables. The shape needs to be specified
        # as well. It it infered automatically for numpy variables.
        # This dict can be nested.
            'alpha': (alpha, ()),
            'beta': (beta, ()),
            'gamma': (gamma, ()),
            'delta': (delta, ()),
            'extra': np.zeros(1),
        },
        # A functions that computes the right-hand-side of the ode using
        # sympy variables.
        rhs=lotka_volterra,
        # The time points where we want to access the solution
        tvals=times,
        t0=times[0],
    )
    
    # We can access the individual variables of the solution using the
    # variable names.
    pm.Deterministic('hares_mu', y_hat['hares'])
    pm.Deterministic('lynx_mu', y_hat['lynx'])
    
    sd = pm.HalfNormal('sd')
    pm.LogNormal('hares', mu=y_hat['hares'], sigma=sd, observed=hare_data)
    pm.LogNormal('lynx', mu=y_hat['lynx'], sigma=sd, observed=lynx_data)


# abstol cant be too low here or there will be machine precision problems with stepsize
lib = sunode._cvodes.lib
lib.CVodeSStolerances(solver._ode, 1e-2,50)
lib.CVodeSStolerancesB(solver._ode, solver._odeB, 1e-3,50)
lib.CVodeQuadSStolerancesB(solver._ode, solver._odeB, 1e-3,50)
# lib.CVodeSetMaxNumSteps(solver._ode, 50000)
# lib.CVodeSetMaxNumStepsB(solver._ode, solver._odeB, 50000)
with model:
    trace = pm.sample(tune=300, draws=300, chains=2, cores=2,target_accept=0.9)

az.summary(trace, round_to=2)
az.summary(trace, kind="stats")
az.plot_trace(trace)
# Posterior parameter distributions
az.plot_posterior(trace, var_names=["period", "ratio", "log_speed_ratio"])
plt.show()
# Posterior predictive checks (built-in)