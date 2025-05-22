


import pymc as pm
import pytensor as pt
from runModel import mosquitoModel, generateData
import parametersDefault
import quantityOfInterest
import odeSolver


parameters = parametersDefault.defaultParameters
initialConditions = parametersDefault.defaultInitialConditions
inferredParameters = ["alpha", "mu_M"]
observedQuantities = [quantityOfInterest.numberOfMosquitos, quantityOfInterest.numberOfEggs]
timeInterval = [0,10]
stepsize = 0.001
ode_Data = generateData(100, timeInterval, parameters, parametersDefault.defaultInitialConditions, 
                        observedQuantities)
print("---------------------------------------------------------")
# odeModel = pm.ode.DifferentialEquation(lambda u,t,p: mosquitoModel(t,u,p).tolist(),
#                                         ode_Data["ts"], n_states=9, n_theta=16)
solver = odeSolver.ODESolver()

with pm.Model() as model:
    #Priors
    alpha = pm.TruncatedNormal("alpha", mu = parameters["alpha"], sigma = 0.5)
    mu_M = pm.TruncatedNormal("mu_M", mu = parameters["mu_M"], sigma = 0.5)
    parameters["alpha"] = alpha
    parameters["mu_M"] = mu_M
    #Forward
    # odeSolution = odeModel(list(parametersDefault.defaultInitialConditions.values()), list(parameters.values()))
    # forwardSolution = []
    # for qoi in quantitiesOfInterest:
    #     forwardSolution.append(qoi(odeSolution[0]))

    
    # Pytensor scan is a looping function
    # solver.solve(mosquitoModel, 10, initialConditions)
    _, interpolantMosquito = solver.solve(lambda t,u: mosquitoModel(t,u,list(parameters.values())), 
                                          timeInterval[1], 
                                          (timeInterval[0], list(initialConditions.values())), 
                                          stepsize, "RK4")
    #Likelihood
    qoi = []
    for qoi in observedQuantities:
        qoi.append(qoi(ode_Data["y"]))
    pm.Normal("Y_obs", mu=qoi, sigma=5000, observed = ode_Data["y"])
    

sampler = "NUTS PyMC ODE"
tune = draws = 15
with model:
    trace_pymc_ode = pm.sample(tune=tune, draws=draws)