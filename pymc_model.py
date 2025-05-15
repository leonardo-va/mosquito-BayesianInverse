


import pymc as pm
from runModel import mosquitoModel, generateData
import parametersDefault
import quantityOfInterest


parameters = parametersDefault.defaultParameters
quantitiesOfInterest = [quantityOfInterest.numberOfMosquitos, quantityOfInterest.numberOfEggs]
ode_Data = generateData(100, [0,10], parameters, parametersDefault.defaultInitialConditions, 
                        quantitiesOfInterest)
print("---------------------------------------------------------")
odeModel = pm.ode.DifferentialEquation(lambda u,t,p: mosquitoModel(t,u,p).tolist(),
                                        ode_Data["ts"], n_states=9, n_theta=16)

with pm.Model() as model:
    #Priors
    alpha = pm.TruncatedNormal("alpha", mu = parameters["alpha"], sigma = 0.5)
    mu_M = pm.TruncatedNormal("mu_M", mu = parameters["mu_M"], sigma = 0.5)
    parameters["alpha"] = alpha
    parameters["mu_M"] = mu_M
    #Forward
    odeSolution = odeModel(list(parametersDefault.defaultInitialConditions.values()), list(parameters.values()))
    print("***********************", odeSolution)
    forwardSolution = []
    for qoi in quantitiesOfInterest:
        forwardSolution.append(qoi(odeSolution[0]))
    #Likelihood
    observation = []
    for qoi in quantitiesOfInterest:
        observation.append(qoi(ode_Data["y"]))
    pm.Normal("Y_obs", mu=odeSolution, sigma=5000, observed = ode_Data["y"])
    print("hi")

sampler = "NUTS PyMC ODE"
tune = draws = 15
with model:
    trace_pymc_ode = pm.sample(tune=tune, draws=draws)