import stan
import parametersDefault
import quantityOfInterest
from runModel import generateData
import time as t

sho = """vector sho(real t,
              vector u,
              real delta_E, real mu_E, real beta, real alpha, real delta_J, real mu_J, real omega, real mu_M,
              real a, real b_M, real alpha_m, real Lambda, real b_H, real mu_H, real alpha_H, real gamma_H) {
      vector[9] res;
      res[1] = beta*(u[3]+u[4]+u[5]) - (delta_E+mu_E)*u[1];
      res[2] = delta_E*u[1] - alpha*(u[2]^2) - mu_J*u[2] - delta_J*u[2];
      res[3] = omega*delta_J*u[2] - a*b_M*u[3]*u[8]/(u[6]+u[7]+u[8]+u[9]) - mu_M*u[3];
      res[4] = a*b_M*u[3]*u[8]/(u[6]+u[7]+u[8]+u[9]) - (alpha_m+mu_M)*u[4];
      res[5] = alpha_m*u[4] - mu_M*u[5];
      res[6] = Lambda - a*b_H*u[5]*u[6]/(u[6]+u[7]+u[8]+u[9]) - mu_H*u[6];
      res[7] = a*b_H*u[5]*u[6]/(u[6]+u[7]+u[8]+u[9]) - alpha_H*u[7] - mu_H*u[7];
      res[8] = alpha_H*u[7]-gamma_H*u[8]-mu_H*u[8];
      res[9] = gamma_H*u[8] - mu_H*u[9];
      return res;
    }"""

qoi = """vector qoi(vector u){
      vector[2] res;
      res[1] = u[3] + u[4] + u[5];
      res[2] = u[1];
      return res;
    }"""

functionsBlock = f"""functions\u007b
{sho}{qoi}\u007d"""

ode_Code = functionsBlock + """
 
  data {
    int<lower=1> T;
    array[T] vector<lower=0>[2] y;
    real t0;
    array[T] real ts;
  }
  parameters {
    real<lower=0> alpha;
    real<lower=0> mu_M;
  }
  model {
    vector[9] y0Mean;
    y0Mean[1] = 10000;
    y0Mean[2] = 10000;
    y0Mean[3] = 50000;
    y0Mean[4] = 10000;
    y0Mean[5] = 10000;
    y0Mean[6] = 10000;
    y0Mean[7] = 1000;
    y0Mean[8] = 1000;
    y0Mean[9] = 0;
    real delta_E = 0.6;
    real mu_E = 0.875;
    real beta = 3;
    real delta_J = 0.09;
    real mu_J = 0.1;
    real omega = 0.5;
    real a = 0.2;
    real b_M = 0.9;
    real alpha_M = 1.0/30;
    real Lambda = 12;
    real b_H = 0.8;
    real mu_H = 0.001;
    real alpha_H = 0.4;
    real gamma_H = 1/5.5;
    real relative_tolerance = 10^(-2);
    real absolute_tolerance = 10^(-2);
    int max_num_steps = 5000;
    array[T] vector[9] mu = ode_rk45_tol(sho, y0Mean, t0, ts, 10^(-2), 10^(-2), 5000 ,delta_E, mu_E, beta, alpha, delta_J, mu_J, omega, mu_M, a, b_M, alpha_M, Lambda, b_H, mu_H, alpha_H, gamma_H);
                                    
  
    alpha ~ normal(10^(-6), 0.5);
    mu_M ~ normal(0.1, 0.5);
  
    // y0 ~ normal(y0Mean, 100);

    for (t in 1:T) {
      vector[2] q = qoi(mu[t]);
      y[t] ~ normal(q, 3000);
    }
  }
"""


# set your data here
ode_Data = generateData(1000, [0,10], parametersDefault.defaultParameters, parametersDefault.defaultInitialConditions, 
                        [quantityOfInterest.numberOfMosquitos, 
                         quantityOfInterest.numberOfEggs])

# build stan model and sample
posterior = stan.build(ode_Code, data=ode_Data)
fit = posterior.sample(num_chains=4, num_samples=1000)


df = fit.to_frame()
df.to_csv(f'samples_{t.time()}.csv')
print(df.describe().T)

 