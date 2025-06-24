import parametersDefault
from odeSolver import ODESolver
import numpy as np
import visualization
from matplotlib import pyplot as plt
import quantityOfInterest
from jsonToModelCode import generate_py_model_function
import json

setup_json_path = 'setup.json'
with open(setup_json_path, 'r') as setup_file:
    setup = json.load(setup_file)


mosquitoModel = generate_py_model_function(setup)

def generateData(N:int, timeInterval: list, parameters: dict, initialCondition: dict, quantitiesOfInterest : list, solverMethod = 'RK4')->dict:
    
    initial = (timeInterval[0], list(initialCondition.values()))
    solver = ODESolver()
    _, interpolantMosquito = solver.solve(lambda t,u: mosquitoModel(t,u,list(parameters.values())), timeInterval[1], initial, 0.01, solverMethod)
    ts = np.linspace(timeInterval[0], timeInterval[1], N)
    us = interpolantMosquito.evalVec(ts)
    combinedQoi = quantitiesOfInterest[0](interpolantMosquito)
    for idx, qoi in enumerate(quantitiesOfInterest):
        if idx == 0:
            continue
        combinedQoi = quantityOfInterest.combine(combinedQoi, qoi(interpolantMosquito))
    us = combinedQoi.evalVec(ts)

    ode_Data = {
        "N": N,
        "y": us.tolist(),
        "t0": ts[0]-0.0001,
        "ts": ts
        }
    return ode_Data


def run():
    alphas = [10**(-6), 2*10**(-6),2.5*10**(-6),3*10**(-6),4*10**(-6),6*10**(-6)]
    parametersList = []
    for alpha in alphas:
        # newParameters = parametersDefault.defaultParameters.copy()
        newParameters = setup['parameters'].copy()
        newParameters["alpha"] = alpha
        parametersList.append(newParameters)
    
    nFigRows = int(np.ceil(len(alphas)/3))
    nFigCols = 3
    figMosquitoPop, axesMosquitoPop = plt.subplots(nFigRows, nFigCols, figsize=(15, 10))
    for idx, params in enumerate(parametersList):
        print(f"alpha = {params['alpha']}")
        _, solutionInterpolant = ODESolver().solve(odeRHS = lambda t,u: mosquitoModel(t,u,
                                                    list(params.values())), 
                                                    T=40,
                                                    initialCondition=(0,np.array(list(setup['initial_state'].values())).reshape(9,-1)),
                                                    stepSize = 0.01,
                                                    method="RK4")
        eggs = quantityOfInterest.eggs(solutionInterpolant)
        juveniles = quantityOfInterest.juveniles(solutionInterpolant)
        total = quantityOfInterest.numberOfMosquitos(solutionInterpolant)
       
        axesMosquitoPop[int(np.floor(idx/nFigCols)),idx % nFigCols].plot(
            eggs.grid, eggs.values, label='Eggs', color='blue')
        axesMosquitoPop[int(np.floor(idx/nFigCols)),idx % nFigCols].plot(
            juveniles.grid, juveniles.values, label='Juveniles', color='green')
        axesMosquitoPop[int(np.floor(idx/nFigCols)),idx % nFigCols].plot(
            total.grid, total.values, label='Total Mosquitos', color='red')
        axesMosquitoPop[int(np.floor(idx/nFigCols)),idx % nFigCols].set_xlabel('Time')
        axesMosquitoPop[int(np.floor(idx/nFigCols)),idx % nFigCols].set_ylabel('Population')
        axesMosquitoPop[int(np.floor(idx/nFigCols)),idx % nFigCols].set_title(
            f'alpha = {params["alpha"]:.3g}')
        axesMosquitoPop[int(np.floor(idx/nFigCols)),idx % nFigCols].legend()
    
    figMosquitoPop.tight_layout()
    figMosquitoPop.show()
    figMosquitoPop.savefig('MosquitoPop.png')
    plt.waitforbuttonpress()

    figSIR, axesSIR = plt.subplots(2,4,figsize=(15,10))
    _, solutionInterpolant = ODESolver().solve(odeRHS = lambda t,u: mosquitoModel(t,u,
                                                list(setup['parameters'].values())), 
                                                T=40,
                                                initialCondition=(0,np.array(list(setup['initial_state'].values())).reshape(9,-1)),
                                                stepSize = 0.01,
                                                method="RK4")
    S_m = quantityOfInterest.S_Mosquitoes(solutionInterpolant)
    E_m = quantityOfInterest.E_Mosquitoes(solutionInterpolant)
    I_m = quantityOfInterest.I_Mosquitoes(solutionInterpolant)
    S_h = quantityOfInterest.S_Hosts(solutionInterpolant)
    E_h = quantityOfInterest.E_Hosts(solutionInterpolant)
    I_h = quantityOfInterest.I_Hosts(solutionInterpolant)
    R_h = quantityOfInterest.R_Hosts(solutionInterpolant)
 
    axesSIR[0,0].plot(S_m.grid, S_m.values, label='Susceptible Mosquitoes', color='blue')
    axesSIR[0,0].legend()
    axesSIR[0,1].plot(E_m.grid, E_m.values, label='Exposed Mosquitoes', color='yellow')
    axesSIR[0,1].legend()
    axesSIR[0,2].plot(I_m.grid, I_m.values, label='Infectious Mosquitoes',color='red')
    axesSIR[0,2].legend()
    axesSIR[0,3].axis('off')

    axesSIR[1,0].plot(S_h.grid, S_h.values, label = 'Susceptible Hosts',color='blue')
    axesSIR[1,0].legend()
    axesSIR[1,1].plot(E_h.grid, E_h.values, label = 'Exposed Hosts',color='yellow')
    axesSIR[1,1].legend()
    axesSIR[1,2].plot(I_h.grid, I_h.values, label = 'Infectious Hosts',color='red')
    axesSIR[1,2].legend()
    axesSIR[1,3].plot(R_h.grid, R_h.values, label = 'Recovered Hosts',color='green')
    axesSIR[1,3].legend()

    figSIR.tight_layout()
    figSIR.show()
    figSIR.savefig('SIR.png')
    plt.waitforbuttonpress()




