import stan
import odeSolver
import numpy as np

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
             vector y,
             real theta,
             real theta2) {
    vector[2] dydt;
    dydt[1] = y[2];
    dydt[2] = -theta2*y[1] - theta * y[2];
    return dydt;
  }
}
data {
  int<lower=1> T;
  array[T] vector[2] y;
  real t0;
  array[T] real ts;
}
parameters {
  vector[2] y0;
  vector<lower=0>[2] sigma;
  real theta;
  real theta2;
}
model {
  array[T] vector[2] mu = ode_rk45(sho, y0, t0, ts, theta, theta2);
  sigma ~ normal(0, 2.5);
  theta ~ std_normal();
  theta2 ~ std_normal();
  y0 ~ std_normal();
  for (t in 1:T) {
    y[t] ~ normal(mu[t], sigma);
  }
}
"""

initial = (1, np.array([0,1]))
theta_Real = [0.5, 0.8]
solver = odeSolver.ODESolver()
_, interpolantOscillator = solver.solve(lambda t,u: harmonicOscillatorEquation(t,u,theta_Real), 10, initial)
ts = np.linspace(1,10,100)
us = interpolantOscillator.evalVec(ts)

ode_Data = {
    "T": 100,
    "y": us.tolist(),
    "t0": 0.5,
    "ts": ts
    }

posterior = stan.build(ode_Code, data=ode_Data)
fit = posterior.sample(num_chains=4, num_samples=100)
print(fit.keys())

df = fit.to_frame()
df.to_csv("harmonic_ocillator_sampling.csv")
print(df.describe().T)