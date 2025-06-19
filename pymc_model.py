


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
    parametersList = list(parameters.values())
    #Forward
    odeRHS = lambda u: mosquitoModel(u, parametersList, modelType="pytensor")
    # Pytensor scan is a looping function
    stepsize = 0.01
    n_steps = timeInterval[1]/stepsize
    print(list(initialConditions.values()))
    # of = [pt.tensor.as_tensor_variable(val) for val in list(initialConditions.values())]
    of = [0,0,0,0,0,0,0,0,0]
    result, updates = pt.scan(
        fn=lambda u0,u1,u2,u3,u4,u5,u6,u7,u8: solver.stepRK4_pymc(u0,u1,u2,u3,u4,u5,u6,u7,u8,odeRHS=odeRHS, stepsize=0.001),  # function
        outputs_info=of,  # initial conditions
        n_steps=n_steps,
    )  # number of loops

    
    
    #Likelihood
    qoi = []
    for qoi in observedQuantities:
        qoi.append(qoi(ode_Data["y"]))
    pm.Normal("Y_obs", mu=qoi, sigma=5000, observed = ode_Data["y"])
    

sampler = "NUTS PyMC ODE"
tune = draws = 15
with model:
    trace_pymc_ode = pm.sample(tune=tune, draws=draws)