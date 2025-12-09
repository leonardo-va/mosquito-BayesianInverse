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
# 1. Define a simple 2D ODE
# -----------------------------
class parametric_class:
    def __init__(self,scaling=1):self.scaling=scaling
    def fct(self,t, fct_params):
        return self.scaling*(fct_params[0]*t + fct_params[1])
    def get(self):
        return self.fct
class ode_maker:
    vals = None
    def __init__(self, vals, fct, fct_params):
        self.vals = vals
        self.fct_params = fct_params
        self.fct = fct
    def simple_ode(self, y, t, theta):
        """
        2D system:
        dy1/dt = a * y1
        dy2/dt = -b * y2
        """
        a=theta[0]
        b1=theta[1]
        b2=theta[2]
        y1, y2 = y
        dy1 = a*self.vals[0] * y2
        dy2 = -self.fct(t,[b1,b2])*self.vals[1] * y1
        return pt.stack([dy1, dy2])
td_fct_c = parametric_class()
td_fct = td_fct_c.get()
maker= ode_maker([4/6,1.2], td_fct,[0.8, 1])
simple_ode = maker.simple_ode

# delta_E_meta = tdp.tempDependentParameterMeta("delta_E", "briere", [3.43, 2.53, 51.69], [1.6*10**4],"egg hatching rate")
# delta_E_tdp = tdp.tempDependentParameter.from_metadata(delta_E_meta)
# delta_E_tdp_pt = tdp.tempDependentParameter.from_metadata(delta_E_meta, pytensor_compatible=True)
# delta_E_t_np = delta_E_tdp.func
# delta_E_t = delta_E_tdp_pt.func

parameters_np, parameters_pt = tdp.make_gt_temp_dependent_parameters(setup)


true_theta_complex = [3.43, 2.53, 51.69]
def complex_ode_np(u,t,theta):
    delta_E = parameters_np['delta_E'].func(t, [theta[0], theta[1],true_theta_complex[2]])
    p_E= 0.875
    beta= 3.0
    alpha= 2e-03
    delta_J= 0.09
    p_J= 0.75
    omega= 0.5
    lf_M= 0.067
    a= 0.2
    b_M= 0.9
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

def complex_ode(u, t, theta):
    # delta_E= 0.6,
    
    delta_E = parameters_pt['delta_E'].func(t, [theta[0], theta[1],true_theta_complex[2]])
    p_E= 0.875
    beta= 3.0
    alpha= 2e-03
    delta_J= 0.09
    p_J= 0.75
    omega= 0.5
    lf_M= 0.067
    a= 0.2
    b_M= 0.9
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
true_theta = np.array([0.5, 0.3,1])
y0 = np.array([1.0, 2.0])
t = np.linspace(0, 5, 20)  # 20 time points

y0_complex = np.array([10000,10000,50000,0,10000,10000,0,0,0])
t_complex = np.linspace(0,100,30)
# Solve ODE using scipy for synthetic observations
true_traj = odeint(lambda y, t: simple_ode(y, t, true_theta).eval(), y0, t) 

true_traj_complex = odeint(lambda y, t: complex_ode_np(y, t, true_theta_complex), y0_complex, t_complex) 
print(true_traj_complex[20], "POSSIBLE IV")
# Add Gaussian observation noise
sigma_obs_true = [10000]
def qoi(traj):
    return traj @ np.array([[1,0],[0,1]]).T
def qoi_pt(traj):
    return traj @ pt.constant(np.array([[1,0],[0,1]]).T)
observed_data = qoi(true_traj) + np.random.normal(0, sigma_obs_true, size=2)
observed_data_complex = true_traj_complex + np.random.normal(0, sigma_obs_true, size=9)
print(true_traj_complex)


# -----------------------------
# 3. Define PyMC model
# -----------------------------
with pm.Model() as model:

    # Priors for ODE parameters
    mu = [3,2]
    sigma = [1,1]
    theta_priors = pm.Normal("theta", mu=mu, sigma=sigma, shape=2)
    # Define the ODE system
    ode_model = pm.ode.DifferentialEquation(
        func=complex_ode,
        times=t_complex,
        n_states=9,
        n_theta=2,
        t0=0
    )

    # Solve the ODE
    y_hat = ode_model(y0=y0_complex, theta=theta_priors)  # shape (20, 2)
    y_hat_transformed = y_hat
    # Likelihood: vectorized
    pm.Normal(
        "y_obs",
        mu=y_hat_transformed,
        sigma=sigma_obs_true[0],
        observed=observed_data_complex
    )

    # -----------------------------
    # 4. Sampling
    # -----------------------------
    trace = pm.sample(20, tune=20, chains=4,cores=4,step=pm.Metropolis())
    pm.Metropolis()
    ppc = pm.sample_posterior_predictive(trace)

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