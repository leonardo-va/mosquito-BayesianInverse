import stan
import parametersDefault
import quantityOfInterest
from runModel import generateData
import time as t

# set your ode code here
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

  vector qoi(vector u){
    vector[2] res;
    res[1] = u[3] + u[4] + u[5];
    res[2] = u[1];
    return res;
  }
}
data {
  int<lower=1> T;
  array[T] vector<lower=0>[2] y;
  real t0;
  array[T] real ts;
}
parameters {
  real<lower=0> p4;
  real<lower=0> p8;
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
  real p1 = 0.6;
  real p2 = 0.875;
  real p3 = 3;
  real p5 = 0.09;
  real p6 = 0.1;
  real p7 = 0.5;
  real p9 = 0.2;
  real p10 = 0.9;
  real p11 = 1.0/30;
  real p12 = 12;
  real p13 = 0.8;
  real p14 = 0.001;
  real p15 = 0.4;
  real p16 = 1/5.5;
  array[T] vector[9] mu = ode_rk45(sho, y0Mean, t0, ts, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16);
 


  p4 ~ normal(10^(-6), 0.5);
  p8 ~ normal(0.1, 0.5);
 
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

  # p1 ~ normal(0.6, 0.1);
  # p2 ~ normal(0.875, 0.1);
  # p3 ~ normal(3, 0.1);
  # p4 ~ normal(10^(-6), 0.00000001);
  # p5 ~ normal(0.09, 0.01);
  # p6 ~ normal(0.1, 0.01);
  # p7 ~ normal(0.5, 0.1);
  # p8 ~ normal(0.1, 0.01);
  # p9 ~ normal(0.2, 0.1);
  # p10 ~ normal(0.9, 0.1);
  # p11 ~ normal(1.0/30, 1.0/60);
  # p12 ~ normal(12, 1);
  # p13 ~ normal(0.8, 0.1);
  # p14 ~ normal(0.001, 0.0001);
  # p15 ~ normal(0.4, 0.1);
  # p16 ~ normal(1/5.5, 0.1);

  # real<lower=0> p1;
  # real<lower=0> p2;
  # real<lower=0> p3;
  # real<lower=0> p4;
  # real<lower=0> p5;
  # real<lower=0> p6;
  # real<lower=0> p7;
  # real<lower=0> p8;
  # real<lower=0> p9;
  # real<lower=0> p10;
  # real<lower=0> p11;
  # real<lower=0> p12;
  # real<lower=0> p13;
  # real<lower=0> p14;
  # real<lower=0> p15;
  # real<lower=0> p16;