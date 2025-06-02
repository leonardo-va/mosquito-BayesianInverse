import stan
import parametersDefault
import quantityOfInterest
from runModel import generateData
import time as t
import json
import jsonToModelCode

setup_json_path = 'setup.json'

with open(setup_json_path, 'r') as setup_file:
    setup = json.load(setup_file)


functionsBlock = jsonToModelCode.generate_stan_function_block(setup)
dataBlock = jsonToModelCode.generate_stan_data_block(setup)
parametersBlock = jsonToModelCode.generate_stan_parameters_block(setup)

ode_Code = functionsBlock + dataBlock + parametersBlock + """
 
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
    array[N] vector[9] mu = ode_rk45_tol(sho, y0Mean, t0, ts, 10^(-2), 10^(-2), 5000 ,delta_E, mu_E, beta, alpha, delta_J, mu_J, omega, mu_M, a, b_M, alpha_M, Lambda, b_H, mu_H, alpha_H, gamma_H);
                                    
  
    alpha ~ normal(10^(-6), 0.5) T[0,1];
    mu_M ~ normal(0.1, 0.5) T[0,1];
  
    // y0 ~ normal(y0Mean, 100);

    for (t in 1:N) {
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

 