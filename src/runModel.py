
from odeSolver import ODESolver
import numpy as np
import os
# import visualization
from matplotlib import pyplot as plt
import quantityOfInterest
from collections import namedtuple

ArtificialData = namedtuple("artificialData", ['noisyData', 'truth', 'standardDeviations'])

def generate_data_from_setup(model_equations, setup:dict):
    numberObservations = setup["number_of_measurements"]
    timeInterval = setup["time_interval"]
    noise = setup["observable_standard_deviation"]
    parameters = setup["parameters"]
    initialState = setup["initial_state"]
    solverMethod = "RK4"
    observables = setup["state_to_observable"]
    quantitiesOfInterest = []
    # print(parameters)
    # printstr = f""
    # for par_func in list(parameters.values()):
    #     timepoints = np.linspace(0,100,10)
    #     printstr += "\n"
    #     for tp in timepoints:
    #         printstr += f"\t {tp}, {par_func(tp)}"
    # print(printstr)
    qoi_names = []
    for observable in setup["state_to_observable"]:
        linearCoefficients = observable["linear_combination"].copy()
        qoi_names.append(observable["name"])
        quantitiesOfInterest.append(lambda interpolant: quantityOfInterest.linearCombinationQOI(interpolant, linearCoefficients))

    initial = (timeInterval[0], list(initialState.values()))
    solver = ODESolver()
    _, interpolantMosquito = solver.solve(lambda t,u: model_equations(t,u,list(parameters.values())), 
                                          timeInterval[1], 
                                          initial, 
                                          0.01, 
                                          solverMethod)

    ts = np.linspace(timeInterval[0], timeInterval[1], numberObservations)
    combinedQoi = quantityOfInterest.linearCombinationQOI(interpolantMosquito, observables[0]["linear_combination"])
    for idx, qoi in enumerate(observables):
        if idx == 0:
            continue
        combinedQoi = quantityOfInterest.combine(combinedQoi, quantityOfInterest.linearCombinationQOI(interpolantMosquito,qoi["linear_combination"]))
    us = combinedQoi.evalVec(ts)
    noisy_data = us.copy()
    for quantity_idx in np.arange(noisy_data.shape[1]):
        noisy_data[:,quantity_idx] += np.random.normal(0,noise[quantity_idx], noisy_data.shape[0])
    noisy_data[noisy_data<0] = 0
       
    ode_Data = {
        "N": numberObservations-1,
        "y": noisy_data[1:,:].tolist(),
        "t0": ts[0],
        "ts": ts[1:]
        }
    groundtruth = {
        "N": numberObservations,
        "y": us.tolist(),
        "t0": ts[0],
        "ts": ts
    }
    return ArtificialData(ode_Data, groundtruth, noise)

def generate_report_plots(mosquitoModel, parameters:dict, initialState:dict, solverMethod = 'RK4', save_png_dir = None):
    
    alphas = [10**(-6), 2*10**(-6),2.5*10**(-6),3*10**(-6),4*10**(-6),6*10**(-6)]
    parametersList = []
    for alpha in alphas:
        newParameters = parameters.copy()
        newParameters["alpha"] = alpha
        parametersList.append(newParameters)
    
    nFigRows = int(np.ceil(len(alphas)/3))
    nFigCols = 3
    figMosquitoPop, axesMosquitoPop = plt.subplots(nFigRows, nFigCols, figsize=(15, 10))
    for idx, params in enumerate(parametersList):
        
        _, solutionInterpolant = ODESolver().solve(odeRHS = lambda t,u: mosquitoModel(t,u,
                                                    list(params.values())), 
                                                    T=100,
                                                    initialCondition=(0,np.array(list(initialState.values())).reshape(9,-1)),
                                                    stepSize = 0.01,
                                                    method=solverMethod)
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
    if(save_png_dir is not None):
        figMosquitoPop.savefig(os.path.join(save_png_dir, 'MosquitoPop.png'))
    plt.waitforbuttonpress()

    figSIR, axesSIR = plt.subplots(2,4,figsize=(15,10))
    _, solutionInterpolant = ODESolver().solve(odeRHS = lambda t,u: mosquitoModel(t,u,
                                                list(parameters.values())), 
                                                T=100,
                                                initialCondition=(0,np.array(list(initialState.values())).reshape(9,-1)),
                                                stepSize = 0.01,
                                                method=solverMethod)
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
    plt.close()

    plt.plot(S_m.grid, S_m.values, label='Susceptible Mosquitoes', color='blue')
    plt.plot(E_m.grid, E_m.values, label='Exposed Mosquitoes', color='orange')
    plt.plot(I_m.grid, I_m.values, label='Infectious Mosquitoes',color='red')
    plt.legend()
    plt.title("Susceptible, Exposed, Infected Mosquitoes")
    plt.savefig(os.path.join(save_png_dir, 'SIR_Mosquitoes.png'))
    plt.show()
  
    
    plt.plot(S_h.grid, S_h.values, label = 'Susceptible Hosts',color='blue')
    plt.plot(E_h.grid, E_h.values, label = 'Exposed Hosts',color='orange')
    plt.plot(I_h.grid, I_h.values, label = 'Infectious Hosts',color='red')
    plt.plot(R_h.grid, R_h.values, label = 'Recovered Hosts',color='green')
    plt.title("Susceptible, Exposed, Infected, Recovered Hosts")
    plt.legend()
    plt.savefig(os.path.join(save_png_dir, 'SIR_Hosts.png'))
    plt.show()
  
    
    
    if(save_png_dir is not None):
        figSIR.savefig(os.path.join(save_png_dir, 'SIR.png'))
    plt.waitforbuttonpress()