import stan
import odeSolver
import numpy as np
import parametersDefault
from runModel import mosquitoModel


ode_Code = """
functions {
  vector sho(real t,
             vector u,
             real delta_E,
             real mu_E,
             real beta,
             real alpha,
             real delta_J,
             real mu_J,
             real omega,
             real mu_M,
             real a,
             real b_M,
             real alpha_M,
             real Lambda,
             real b_H,
             real mu_H,
             real alpha_H,
             real gamma_H) {
    
    real e = u[1];
    real j = u[2];
    real sm = u[3];
    real em = u[4];
    real im = u[5];
    real sh = u[6];
    real eh = u[7];
    real ih = u[8];
    real rh = u[9];

    real m = sm + em + im;
    real nh = sh + eh + ih + rh;

    vector[9] dudt;

    dudt[1] = beta*m - (delta_E+mu_E)*e;
    dudt[2] = delta_E*e - alpha*(j*j) - mu_J*j - delta_J*j;
    dudt[3] = omega*delta_J*j - a*b_M*sm*ih/nh - mu_M*sm;
    dudt[4] = a*b_M*sm*ih/nh - (alpha_M+mu_M)*em;
    dudt[5] = alpha_M*em - mu_M*im;
    dudt[6] = Lambda - a*b_H*im*sh/nh - mu_H*sh;
    dudt[7] = a*b_H*im*sh/nh - alpha_H*eh - mu_H*eh;
    dudt[8] = alpha_H*eh-gamma_H*ih-mu_H*ih;
    dudt[9] = gamma_H*ih - mu_H*rh;

    return dudt;
  }
}
data {
  int<lower=1> T;
  array[T] vector[9] y;
  real t0;
  array[T] real ts;
}
parameters {
  vector[9]<lower=0> y0;
  vector<lower=0>[9] sigma;
  real delta_E;
  real mu_E;
  real beta;
  real alpha;
  real delta_J;
  real mu_J;
  real omega;
  real mu_M;
  real a;
  real b_M;
  real alpha_M;
  real Lambda;
  real b_H;
  real mu_H;
  real alpha_H;
  real gamma_H;
}
model {
  array[T] vector[9] mu = ode_rk45(sho, y0, t0, ts, delta_E, mu_E, beta, alpha, delta_J, mu_J, omega, mu_M, a, b_M, alpha_M, Lambda, b_H, mu_H, alpha_H, gamma_H);
  delta_E ~ std_normal();
  mu_E ~ std_normal();
  beta ~ std_normal();
  alpha ~ std_normal();
  delta_J ~ std_normal();
  mu_J ~ std_normal();
  omega ~ std_normal();
  mu_M ~ std_normal();
  a ~ std_normal();
  b_M ~ std_normal();
  alpha_M ~ std_normal();
  Lambda ~ std_normal();
  b_H ~ std_normal();
  mu_H ~ std_normal();
  alpha_H ~ std_normal();
  gamma_H ~ std_normal();
  y0 ~ std_normal();
  for (t in 1:T) {
    y[t] ~ normal(mu[t], 3000);
  }
}
"""

initial = (0,np.array(list(parametersDefault.initialConditions.values())).reshape(9,-1))
params_Real = list(parametersDefault.parameters.values())
solver = odeSolver.ODESolver()
_, interpolantMosquitoModel = solver.solve(lambda t,u: mosquitoModel(t,u,params_Real), 10, initial)
ts = np.linspace(1,10,100)
us = interpolantMosquitoModel.evalVec(ts)

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