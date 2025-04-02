import stan
import odeSolver
import numpy as np
import parametersDefault
from runModel import mosquitoModel

def harmonicOscillatorEquation(t,u,parameters):

    # dudt = np.array([0,0])
    u = u.T
    dudt = np.zeros(u.shape)
    theta = parameters[0]
    dudt[0] = u[1]
    dudt[1] = -u[0] - theta * u[1]
    return dudt.T

ode_Code = """
functions {
  vector sho(real t,
            vector u,
            vector model_params) {
    vector[9] res;
    res[1] = model_params[3]*(u[3]+u[4]+u[5]) - (model_params[1]+model_params[2])*u[1];
    res[2] = model_params[1]*u[1] - model_params[4]*(u[2]^2) - model_params[6]*u[2] - model_params[5]*u[2];
    res[3] = model_params[7]*model_params[5]*u[2] - model_params[9]*model_params[10]*u[3]*u[8]/(u[6]+u[7]+u[8]+u[9]) - model_params[8]*u[3];
    res[4] = model_params[9]*model_params[10]*u[3]*u[8]/(u[6]+u[7]+u[8]+u[9]) - (model_params[11]+model_params[8])*u[4];
    res[5] = model_params[11]*u[4] - model_params[8]*u[5];
    res[6] = model_params[12] - model_params[9]*model_params[13]*u[5]*u[6]/(u[6]+u[7]+u[8]+u[9]) - model_params[14]*u[6];
    res[7] = model_params[9]*model_params[13]*u[5]*u[6]/(u[6]+u[7]+u[8]+u[9]) - model_params[15]*u[7] - model_params[14]*u[7];
    res[8] = model_params[15]*u[7]-model_params[16]*u[8]-model_params[14]*u[8];
    res[9] = model_params[16]*u[8] - model_params[14]*u[9];
    return res;
  }
}
data {
  int<lower=1> T;
  array[T] vector<lower=0>[9] y;
  real t0;
  array[T] real ts;
}
parameters {
  vector<lower=0>[9] y0;
  vector<lower=0>[9] sigma;
  vector<lower=0>[16] model_params;
}
model {
  array[T] vector[9] mu = ode_rk45(sho, y0, t0, ts, model_params);
  sigma ~ normal(0, 250);

  model_params ~ normal(0.5, 0.5);

  y0 ~ normal(10000, sigma);
  for (t in 1:T) {
    y[t] ~ normal(mu[t], sigma);
  }
}
"""

initial = (0, list(parametersDefault.initialConditions.values()))
parameters_Real = list(parametersDefault.parameters.values())
solver = odeSolver.ODESolver()
_, interpolantMosquito = solver.solve(lambda t,u: mosquitoModel(t,u,parameters_Real), 40, initial)
ts = np.linspace(1,35,10000)
us = interpolantMosquito.evalVec(ts)

ode_Data = {
    "T": 10000,
    "y": us.tolist(),
    "t0": 0,
    "ts": ts
    }

posterior = stan.build(ode_Code, data=ode_Data)
fit = posterior.sample(num_chains=4, num_samples=100)
print(fit.keys())

df = fit.to_frame()
df.to_csv("harmonic_ocillator_sampling.csv")
print(df.describe().T)

#   model_params[1] ~ normal(0.6, 0.1);
#   model_params[2] ~ normal(0.875, 0.1);
#   model_params[3] ~ normal(3, 0.1);
#   model_params[4] ~ normal(10^(-6), 0.00000001);
#   model_params[5] ~ normal(0.09, 0.01);
#   model_params[6] ~ normal(0.1, 0.01);
#   model_params[7] ~ normal(0.5, 0.1);
#   model_params[8] ~ normal(0.1, 0.01);
#   model_params[9] ~ normal(0.2, 0.1);
#   model_params[10] ~ normal(0.9, 0.1);
#   model_params[11] ~ normal(1.0/30, 1.0/60);
#   model_params[12] ~ normal(12, 1);
#   model_params[13] ~ normal(0.8, 0.1);
#   model_params[14] ~ normal(0.001, 0.0001);
#   model_params[15] ~ normal(0.4, 0.1);
#   model_params[16] ~ normal(1/5.5, 0.1);