import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import arviz as az
import pytensor
import pytensor.printing as ptp
import temperature_dependent_parameters as tdp
import os
import json
from jsonToModelCode import validate_setup
import argparse
import sunode
from time import perf_counter
from sunode.wrappers.as_pytensor import solve_ivp


def _get_root_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    return parent_dir

def _get_setup_path():
    root_dir = _get_root_dir()
    default_setup_path = os.path.join(root_dir, "setup.json")
    return default_setup_path

def pt_debugger(item, msg = "DEBUGMSG"):
    debug = ptp.Print(msg)(item)
    f = pytensor.function([], debug)
    f() 

def get_setup():
    setup_path_default_arg = _get_setup_path()
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", default = setup_path_default_arg, type=str, help="Path to the setup.json file")
    args = parser.parse_args()
    with open(args.setup, 'r') as setup_file:
        setup = json.load(setup_file)

    if validate_setup(setup) == False:
        print("setup is not valid")
        return None
    return setup
setup = get_setup() 
  
# -----------------------------
# 1. Define ODE
# -----------------------------

# delta_E_meta = tdp.tempDependentParameterMeta("delta_E", "briere", [3.43, 2.53, 51.69], [1.6*10**4],"egg hatching rate")
# delta_E_tdp = tdp.tempDependentParameter.from_metadata(delta_E_meta)
# delta_E_tdp_pt = tdp.tempDependentParameter.from_metadata(delta_E_meta, pytensor_compatible=True)
# delta_E_t_np = delta_E_tdp.func
# delta_E_t = delta_E_tdp_pt.func

parameters_np, parameters_pt = tdp.make_gt_temp_dependent_parameters(setup)

true_theta_temp_dependent = [3.43, 2.53, 51.69]
true_theta_temp_constant = [2*10**(-6), 0.067]
def ode_np(u,t,theta):
    temp = tdp.temperature_profile(t)
    delta_E = parameters_np['delta_E'].func(temp, [theta[0], theta[1],true_theta_temp_dependent[2]])
    p_E= parameters_np['p_E'].func(temp, parameters_np['p_E'].get_function_parameters())
    beta= parameters_np['beta'].func(temp, parameters_np['beta'].get_function_parameters())
    alpha= parameters_np['alpha'].func(temp, parameters_np['alpha'].get_function_parameters())
    delta_J= parameters_np['delta_J'].func(temp, parameters_np['delta_J'].get_function_parameters())
    p_J= parameters_np['p_J'].func(temp, parameters_np['p_J'].get_function_parameters())
    omega= parameters_np['omega'].func(temp, parameters_np['omega'].get_function_parameters())
    lf_M= parameters_np['lf_M'].func(temp, parameters_np['lf_M'].get_function_parameters())
    a= parameters_np['a'].func(temp, parameters_np['a'].get_function_parameters())
    b_M = parameters_np['b_M'].func(temp, parameters_np['b_M'].get_function_parameters())
    EIP_M= parameters_np['EIP_M'].func(temp, parameters_np['EIP_M'].get_function_parameters())
    Lambda= parameters_np['Lambda'].func(temp, parameters_np['Lambda'].get_function_parameters())
    b_H= parameters_np['b_H'].func(temp, parameters_np['b_H'].get_function_parameters())
    mu_H= parameters_np['mu_H'].func(temp, parameters_np['mu_H'].get_function_parameters())
    alpha_H= parameters_np['alpha_H'].func(temp, parameters_np['alpha_H'].get_function_parameters())
    gamma_H= parameters_np['gamma_H'].func(temp, parameters_np['gamma_H'].get_function_parameters())
    r1 = beta*(u[2]+u[3]+u[4]) - delta_E * u[0] - (delta_E/p_E - delta_E) * u[0]
    r2 = delta_E * u[0] - delta_J * u[1] -alpha*u[1]**2 - (delta_J/p_J - delta_J) * u[1]
    r3 = omega*delta_J * u[1] - a*b_M*u[2]*u[7]/(u[5]+u[6]+u[7]+u[8]) - 1/(lf_M*150)*u[2]
    r4 = a*b_M*u[2]*u[7]/(u[5]+u[6]+u[7]+u[8]) - 1/EIP_M*u[3] - 1/(lf_M*150)*u[3]
    r5 = 1/EIP_M*u[3] - 1/(lf_M*150)*u[4]
    r6 = Lambda - a*b_H*u[5]*u[4]/(u[2]+u[3]+u[4]) - mu_H*u[5]
    r7 = a*b_H*u[5]*u[4]/(u[2]+u[3]+u[4]) - alpha_H*u[6] - mu_H*u[6]
    r8 = alpha_H*u[6] - gamma_H*u[7] - mu_H*u[7]
    r9 = gamma_H*u[7] - mu_H * u[8]
    return np.array([r1,r2,r3,r4,r5,r6,r7,r8,r9])

def ode_pt(u, t, theta):
    temp = tdp.temperature_profile_pt(t)
    delta_E = parameters_pt['delta_E'].func(temp, [theta[0], theta[1],true_theta_temp_dependent[2]])
    p_E= parameters_pt['p_E'].func(temp, parameters_pt['p_E'].get_function_parameters())
    beta= parameters_pt['beta'].func(temp, parameters_pt['beta'].get_function_parameters())
    alpha= parameters_pt['alpha'].func(temp, parameters_pt['alpha'].get_function_parameters())
    delta_J= parameters_pt['delta_J'].func(temp, parameters_pt['delta_J'].get_function_parameters())
    p_J= parameters_pt['p_J'].func(temp, parameters_pt['p_J'].get_function_parameters())
    omega= parameters_pt['omega'].func(temp, parameters_pt['omega'].get_function_parameters())
    lf_M= parameters_pt['lf_M'].func(temp, parameters_pt['lf_M'].get_function_parameters())
    a= parameters_pt['a'].func(temp, parameters_pt['a'].get_function_parameters())
    b_M = parameters_pt['b_M'].func(temp, parameters_pt['b_M'].get_function_parameters())
    EIP_M= parameters_pt['EIP_M'].func(temp, parameters_pt['EIP_M'].get_function_parameters())
    Lambda= parameters_pt['Lambda'].func(temp, parameters_pt['Lambda'].get_function_parameters())
    b_H= parameters_pt['b_H'].func(temp, parameters_pt['b_H'].get_function_parameters())
    mu_H= parameters_pt['mu_H'].func(temp, parameters_pt['mu_H'].get_function_parameters())
    alpha_H= parameters_pt['alpha_H'].func(temp, parameters_pt['alpha_H'].get_function_parameters())
    gamma_H= parameters_pt['gamma_H'].func(temp, parameters_pt['gamma_H'].get_function_parameters())
    r1 = beta*(u[2]+u[3]+u[4]) - delta_E * u[0] - (delta_E/p_E - delta_E) * u[0]
    r2 = delta_E * u[0] - delta_J * u[1] -alpha*u[1]**2 - (delta_J/p_J - delta_J) * u[1]
    r3 = omega*delta_J * u[1] - a*b_M*u[2]*u[7]/(u[5]+u[6]+u[7]+u[8]) - 1/(lf_M*150)*u[2]
    r4 = a*b_M*u[2]*u[7]/(u[5]+u[6]+u[7]+u[8]) - 1/EIP_M*u[3] - 1/(lf_M*150)*u[3]
    r5 = 1/EIP_M*u[3] - 1/(lf_M*150)*u[4]
    r6 = Lambda - a*b_H*u[5]*u[4]/(u[2]+u[3]+u[4]) - mu_H*u[5]
    r7 = a*b_H*u[5]*u[4]/(u[2]+u[3]+u[4]) - alpha_H*u[6] - mu_H*u[6]
    r8 = alpha_H*u[6] - gamma_H*u[7] - mu_H*u[7]
    r9 = gamma_H*u[7] - mu_H * u[8]
    return pt.stack([r1,r2,r3,r4,r5,r6,r7,r8,r9])

def ode_sun(t,u, theta):
    temp = tdp.temperature_profile(t)
    delta_E = parameters_pt['delta_E'].func(temp, [theta[0], theta[1],true_theta_temp_dependent[2]])
    p_E= parameters_pt['p_E'].func(temp, parameters_pt['p_E'].get_function_parameters())
    beta= parameters_pt['beta'].func(temp, parameters_pt['beta'].get_function_parameters())
    alpha= parameters_pt['alpha'].func(temp, parameters_pt['alpha'].get_function_parameters())
    delta_J= parameters_pt['delta_J'].func(temp, parameters_pt['delta_J'].get_function_parameters())
    p_J= parameters_pt['p_J'].func(temp, parameters_pt['p_J'].get_function_parameters())
    omega= parameters_pt['omega'].func(temp, parameters_pt['omega'].get_function_parameters())
    lf_M= parameters_pt['lf_M'].func(temp, parameters_pt['lf_M'].get_function_parameters())
    a= parameters_pt['a'].func(temp, parameters_pt['a'].get_function_parameters())
    b_M = parameters_pt['b_M'].func(temp, parameters_pt['b_M'].get_function_parameters())
    EIP_M= parameters_pt['EIP_M'].func(temp, parameters_pt['EIP_M'].get_function_parameters())
    Lambda= parameters_pt['Lambda'].func(temp, parameters_pt['Lambda'].get_function_parameters())
    b_H= parameters_pt['b_H'].func(temp, parameters_pt['b_H'].get_function_parameters())
    mu_H= parameters_pt['mu_H'].func(temp, parameters_pt['mu_H'].get_function_parameters())
    alpha_H= parameters_pt['alpha_H'].func(temp, parameters_pt['alpha_H'].get_function_parameters())
    gamma_H= parameters_pt['gamma_H'].func(temp, parameters_pt['gamma_H'].get_function_parameters())
    return{"r1" : beta*(u[2]+u[3]+u[4]) - delta_E * u[0] - (delta_E/p_E - delta_E) * u[0],
    "r2" : delta_E * u[0] - delta_J * u[1] -alpha*u[1]**2 - (delta_J/p_J - delta_J) * u[1],
    "r3" : omega*delta_J * u[1] - a*b_M*u[2]*u[7]/(u[5]+u[6]+u[7]+u[8]) - 1/(lf_M*150)*u[2],
    "r4" : a*b_M*u[2]*u[7]/(u[5]+u[6]+u[7]+u[8]) - 1/EIP_M*u[3] - 1/(lf_M*150)*u[3],
    "r5" : 1/EIP_M*u[3] - 1/(lf_M*150)*u[4],
    "r6" : Lambda - a*b_H*u[5]*u[4]/(u[2]+u[3]+u[4]) - mu_H*u[5],
    "r7" : a*b_H*u[5]*u[4]/(u[2]+u[3]+u[4]) - alpha_H*u[6] - mu_H*u[6],
    "r8" : alpha_H*u[6] - gamma_H*u[7] - mu_H*u[7],
    "r9" : gamma_H*u[7] - mu_H * u[8]}

def ode_temp_independent_np(u,t,theta):
    delta_E =0.6
    p_E= 0.875
    beta= 3
    alpha= theta[0]
    delta_J= 0.09
    p_J= 0.75
    omega= 0.5
    lf_M= theta[1]
    a= 0.2
    b_M = 0.9
    EIP_M= 30
    Lambda= 12
    b_H= 0.8
    mu_H= 0.001
    alpha_H= 0.4
    gamma_H= 0.1818
    r1 = beta*(u[2]+u[3]+u[4]) - delta_E * u[0] - (delta_E/p_E - delta_E) * u[0]
    r2 = delta_E * u[0] - delta_J * u[1] -alpha*u[1]**2 - (delta_J/p_J - delta_J) * u[1]
    r3 = omega*delta_J * u[1] - a*b_M*u[2]*u[7]/(u[5]+u[6]+u[7]+u[8]) - 1/(lf_M*150)*u[2]
    r4 = a*b_M*u[2]*u[7]/(u[5]+u[6]+u[7]+u[8]) - 1/EIP_M*u[3] - 1/(lf_M*150)*u[3]
    r5 = 1/EIP_M*u[3] - 1/(lf_M*150)*u[4]
    r6 = Lambda - a*b_H*u[5]*u[4]/(u[2]+u[3]+u[4]) - mu_H*u[5]
    r7 = a*b_H*u[5]*u[4]/(u[2]+u[3]+u[4]) - alpha_H*u[6] - mu_H*u[6]
    r8 = alpha_H*u[6] - gamma_H*u[7] - mu_H*u[7]
    r9 = gamma_H*u[7] - mu_H * u[8]
    return np.array([r1,r2,r3,r4,r5,r6,r7,r8,r9])
def ode_temp_independent_pt(u,t,theta):
    delta_E =0.6
    p_E= 0.875
    beta= 3
    alpha= theta[0]
    delta_J= 0.09
    p_J= 0.75
    omega= 0.5
    lf_M= theta[1]
    a= 0.2
    b_M = 0.9
    EIP_M= 30
    Lambda= 12
    b_H= 0.8
    mu_H= 0.001
    alpha_H= 0.4
    gamma_H= 0.1818
    r1 = beta*(u[2]+u[3]+u[4]) - delta_E * u[0] - (delta_E/p_E - delta_E) * u[0]
    r2 = delta_E * u[0] - delta_J * u[1] -alpha*u[1]**2 - (delta_J/p_J - delta_J) * u[1]
    r3 = omega*delta_J * u[1] - a*b_M*u[2]*u[7]/(u[5]+u[6]+u[7]+u[8]) - 1/(lf_M*150)*u[2]
    r4 = a*b_M*u[2]*u[7]/(u[5]+u[6]+u[7]+u[8]) - 1/EIP_M*u[3] - 1/(lf_M*150)*u[3]
    r5 = 1/EIP_M*u[3] - 1/(lf_M*150)*u[4]
    r6 = Lambda - a*b_H*u[5]*u[4]/(u[2]+u[3]+u[4]) - mu_H*u[5]
    r7 = a*b_H*u[5]*u[4]/(u[2]+u[3]+u[4]) - alpha_H*u[6] - mu_H*u[6]
    r8 = alpha_H*u[6] - gamma_H*u[7] - mu_H*u[7]
    r9 = gamma_H*u[7] - mu_H * u[8]
    return pt.stack([r1,r2,r3,r4,r5,r6,r7,r8,r9])
# -----------------------------
# 2. Generate synthetic data
# -----------------------------
# y_0 = np.array([10000,10000,50000,0,10000,10000,0,0,0])
y_0 = np.array([850000,  480000,  200000,   1000,
    1000,  10000,    0,   0, 0])
ts = np.linspace(0,100,30)
sigma_obs_true = [15000,7000]
n_states = 9
n_observables = 2
n_inferred_params = 2
prior_means = [0.5,0.5]
prior_sigma = [0.1,0.1]
# Solve ODE using scipy for synthetic observations
solve_start = perf_counter()
true_traj = odeint(lambda y, t: ode_temp_independent_np(y, t, true_theta_temp_constant), y_0, ts) 
solve_end = perf_counter()
print("Time for 1 solve:", solve_end-solve_start)
print(true_traj[20], "POSSIBLE IV")
# Add Gaussian observation noise
def qoi(traj):
    return traj @ np.array([[0,1,0,0,0,0,0,0,0],[0,0,1,1,1,0,0,0,0]]).T
def qoi_pt(traj):
    return traj @ pt.constant(np.array([[0,1,0,0,0,0,0,0,0],[0,0,1,1,1,0,0,0,0]])).T
observed_data = qoi(true_traj) + np.random.normal(0, sigma_obs_true, size=n_observables)
print(observed_data)
print(true_traj)


# -----------------------------
# 3. Define PyMC model
# -----------------------------
with pm.Model() as model:

    # Priors for ODE parameters
    # mu = [3,2]
    # sigma = [1,1]
    # theta_priors = pm.Normal("theta", mu=prior_means, sigma=prior_sigma, shape=2)
    theta_priors = pm.HalfNormal("theta", sigma=prior_sigma,shape=2)
    # alpha_prior = pm.HalfNormal("alpha", sigma = 0.1)

    # Define the ODE system
    ode_model = pm.ode.DifferentialEquation(
        func=ode_pt,
        times=ts,
        n_states=n_states,
        n_theta=n_inferred_params,
        t0=0
    )

    # Solve the ODE
    y_hat = ode_model(y0=y_0, theta=theta_priors)  # shape (20, 2)
    y_hat_transformed = qoi_pt(y_hat)
    # Likelihood: vectorized
    pt_debugger(y_hat_transformed.shape)
    
    pm.Normal(
        "y_obs",
        mu=y_hat_transformed,
        sigma=sigma_obs_true,
        observed=observed_data
    )

    # -----------------------------
    # 4. Sampling
    # -----------------------------
    trace = pm.sample(20, tune=20, chains=4,cores=4, start={"theta":[0.0002, 0.067]})
    # pm.Metropolis()
    ppc = pm.sample_posterior_predictive(trace)
model.debug()
print(trace)

# -----------------------------
# 4. ArviZ Visualizations
# -----------------------------
az.summary(trace, round_to=2)
az.summary(trace, kind="stats")
az.plot_trace(trace)
# Posterior parameter distributions
az.plot_posterior(trace, var_names=["theta"])
plt.show()
# Posterior predictive checks (built-in)