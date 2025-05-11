


import pymc as pm
from runModel import mosquitoModel, generateData
import parametersDefault
import quantityOfInterest


parameters = parametersDefault.defaultParameters
ode_Data = generateData(100, [0,10], parameters, parametersDefault.defaultInitialConditions, 
                        [quantityOfInterest.numberOfMosquitos, 
                         quantityOfInterest.numberOfEggs])
print("---------------------------------------------------------")
odeModel = pm.ode.DifferentialEquation(lambda u,t,p: mosquitoModel(t,u,p).tolist(), ode_Data["ts"], n_states=9, n_theta=16)
pm.TruncatedNormal()
with pm.Model() as model:
    #Priors
    alpha = pm.TruncatedNormal("alpha", mu = parameters["alpha"], sigma = 0.5)
    mu_M = pm.TruncatedNormal("alpha", mu = parameters["mu_M"], sigma = 0.5)