import stan
import odeSolver
import numpy as np

def sirModel(t, u, parameters):
    u = u.T
    S = u[0]
    I = u[1]
    R = u[2]
    N = S+I+R
    dudt = np.zeros(u.shape)
    nu = parameters[0]
    beta = parameters[1]
    mu = parameters[2]
    gamma = parameters[3]
    dudt[0] = nu*N - beta * S*I/N - mu*S
    dudt[1] = beta * S*I/N - gamma*I - mu*I
    dudt[2] = gamma*I - mu*R
    return dudt.T


ode_Code = """
functions {
  vector sho(real t,
             vector y,
             real nu,
             real beta,
             real mu,
             real gamma) {
    vector[3] dydt;
    real N = y[1] + y[2] + y[3];
    real S = y[1];
    real I = y[2];
    real R = y[3];
    dydt[1] = nu*N - beta * S*I/N - mu*S;
    dydt[2] = beta * S*I/N - gamma*I - mu*I;
    dydt[3] = gamma*I - mu*R;
    return dydt;
  }
}
data {
  int<lower=1> T;
  array[T] vector[3] y;
  real t0;
  array[T] real ts;
}
parameters {
  vector[3] y0;
  vector<lower=0>[3] sigma;
  real nu;
  real beta;
  real mu;
  real gamma;
}
model {
  array[T] vector[3] forward = ode_rk45(sho, y0, t0, ts, nu,beta,mu,gamma);
  sigma ~ normal(0, 2.5);
  nu ~ std_normal();
  beta ~ std_normal();
  mu ~ std_normal();
  gamma ~ std_normal();
  y0 ~ std_normal();
  for (t in 1:T) {
    y[t] ~ normal(forward[t], sigma);
  }
}
"""

initial = (0, np.array([1000,0,0]))
theta_Real = [0.1, 0.1, 0.1, 0.1]
solver = odeSolver.ODESolver()
_, interpolantOscillator = solver.solve(lambda t,u: sirModel(t,u,theta_Real), 10, initial)
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