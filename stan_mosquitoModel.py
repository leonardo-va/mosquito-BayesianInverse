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
            real p1, real p2, real p3, real p4, real p5, real p6, real p7, real p8,
            real p9, real p10, real p11, real p12, real p13, real p14, real p15, real p16) {
    vector[9] res;
    res[1] = p3*(u[3]+u[4]+u[5]) - (p1+p2)*u[1];
    res[2] = p1*u[1] - p4*(u[2]^2) - p6*u[2] - p5*u[2];
    res[3] = p7*p5*u[2] - p9*p10*u[3]*u[8]/(u[6]+u[7]+u[8]+u[9]) - p8*u[3];
    res[4] = p9*p10*u[3]*u[8]/(u[6]+u[7]+u[8]+u[9]) - (p11+p8)*u[4];
    res[5] = p11*u[4] - p8*u[5];
    res[6] = p12 - p9*p13*u[5]*u[6]/(u[6]+u[7]+u[8]+u[9]) - p14*u[6];
    res[7] = p9*p13*u[5]*u[6]/(u[6]+u[7]+u[8]+u[9]) - p15*u[7] - p14*u[7];
    res[8] = p15*u[7]-p16*u[8]-p14*u[8];
    res[9] = p16*u[8] - p14*u[9];
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
  real<lower=0> p1;
  real<lower=0> p2;
  real<lower=0> p3;
  real<lower=0> p4;
  real<lower=0> p5;
  real<lower=0> p6;
  real<lower=0> p7;
  real<lower=0> p8;
  real<lower=0> p9;
  real<lower=0> p10;
  real<lower=0> p11;
  real<lower=0> p12;
  real<lower=0> p13;
  real<lower=0> p14;
  real<lower=0> p15;
  real<lower=0> p16;
}
model {
  array[T] vector[9] mu = ode_rk45(sho, y0, t0, ts, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16);
  sigma ~ normal(0, 250);

  p1 ~ normal(0.6, 0.1);
  p2 ~ normal(0.875, 0.1);
  p3 ~ normal(3, 0.1);
  p4 ~ normal(10^(-6), 0.00000001);
  p5 ~ normal(0.09, 0.01);
  p6 ~ normal(0.1, 0.01);
  p7 ~ normal(0.5, 0.1);
  p8 ~ normal(0.1, 0.01);
  p9 ~ normal(0.2, 0.1);
  p10 ~ normal(0.9, 0.1);
  p11 ~ normal(1.0/30, 1.0/60);
  p12 ~ normal(12, 1);
  p13 ~ normal(0.8, 0.1);
  p14 ~ normal(0.001, 0.0001);
  p15 ~ normal(0.4, 0.1);
  p16 ~ normal(1/5.5, 0.1);

  y0 ~ normal(10000, sigma);
  for (t in 1:T) {
    y[t] ~ normal(mu[t], sigma);
  }
  
}
"""

initial = (0, list(parametersDefault.initialConditions.values()))
parameters_Real = list(parametersDefault.parameters.values())
solver = odeSolver.ODESolver()
_, interpolantMosquito = solver.solve(lambda t,u: mosquitoModel(t,u,parameters_Real), 5, initial)
ts = np.linspace(1,5,100)
us = interpolantMosquito.evalVec(ts)

ode_Data = {
    "T": 100,
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