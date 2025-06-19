import parametersDefault
from odeSolver import ODESolver, stepRK4_pymc
import numpy as np
import visualization
from matplotlib import pyplot as plt
import quantityOfInterest
import pytensor.tensor as pt

def mosquitoModel(u, parameters, modelType="pytensor"):
    '''
    Forward model:

    ODE descrbing the population dynamics: u'(t) = f(t,u(t),parameters)

    return: u'  

    As model type, use either pytensor or numpy
    '''
    
    res0 = parameters[2]*(u[2]+u[3]+u[4]) - (parameters[0]+parameters[1])*u[0]
    res1 = parameters[0]*u[0] - parameters[3]*(u[1]**2) - parameters[5]*u[1] - parameters[4]*u[1]
    res2 = parameters[6]*parameters[4]*u[1] - parameters[8]*parameters[9]*u[2]*u[7]/(u[5]+u[6]+u[7]+u[8]) - parameters[7]*u[2]
    res3 = parameters[8]*parameters[9]*u[2]*u[7]/(u[5]+u[6]+u[7]+u[8]) - (parameters[10]+parameters[7])*u[3]
    res4 = parameters[10]*u[3] - parameters[7]*u[4]
    res5 = parameters[11] - parameters[8]*parameters[12]*u[4]*u[5]/(u[5]+u[6]+u[7]+u[8]) - parameters[13]*u[5]
    res6 = parameters[8]*parameters[12]*u[4]*u[5]/(u[5]+u[6]+u[7]+u[8]) - parameters[14]*u[6] - parameters[13]*u[6]
    res7 = parameters[14]*u[6]-parameters[15]*u[7]-parameters[13]*u[7]
    res8 = parameters[15]*u[7] - parameters[13]*u[8]
    
    if(modelType=="pytensor"):
        # return pt.stack([res0, res1, res2, res3, res4, res5, res6, res7, res8])
        return res0,res1,res2,res3,res4,res5,res6,res7,res8
    else:
        return np.array([res0, res1, res2, res3, res4, res5, res6, res7, res8])
    

def generateData(N:int, timeInterval: list, parameters: dict, initialCondition: dict, quantitiesOfInterest : list, solverMethod = 'RK4')->dict:
    
    initial = (timeInterval[0], np.array(list(initialCondition.values())))
    solver = ODESolver()
    _, interpolantMosquito = solver.solve(lambda u: mosquitoModel(u,list(parameters.values()), modelType="numpy"), timeInterval[1], initial, 0.001, solverMethod)
    ts = np.linspace(timeInterval[0], timeInterval[1], N)
    us = interpolantMosquito.evalVec(ts)
    combinedQoi = quantitiesOfInterest[0](interpolantMosquito)
    for idx, qoi in enumerate(quantitiesOfInterest):
        if idx == 0:
            continue
        combinedQoi = quantityOfInterest.combine(combinedQoi, qoi(interpolantMosquito))
    us = combinedQoi.evalVec(ts)

    ode_Data = {
        "T": N,
        "y": us.tolist(),
        "t0": ts[0]-0.0001,
        "ts": ts
        }
    return ode_Data


def run():
    alphas = [10**(-6), 2*10**(-6),2.5*10**(-6),3*10**(-6),4*10**(-6),6*10**(-6)]
    parametersList = []
    for alpha in alphas:
        newParameters = parametersDefault.defaultParameters.copy()
        newParameters["alpha"] = alpha
        parametersList.append(newParameters)
    
    nFigRows = int(np.ceil(len(alphas)/3))
    nFigCols = 3
    figMosquitoPop, axesMosquitoPop = plt.subplots(nFigRows, nFigCols, figsize=(15, 10))
    for idx, params in enumerate(parametersList):
        print(f"alpha = {params['alpha']}")
        _, solutionInterpolant = ODESolver().solve(odeRHS = lambda u: mosquitoModel(u,
                                                    list(params.values()), modelType="numpy"), 
                                                    T=40,
                                                    initialCondition=(0,np.array(list(parametersDefault.defaultInitialConditions.values())).reshape(9,-1)),
                                                    stepSize = 0.001,
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
    _, solutionInterpolant = ODESolver().solve(odeRHS = lambda u: mosquitoModel(u,
                                                list(parametersDefault.defaultParameters.values()), modelType="numpy"), 
                                                T=40,
                                                initialCondition=(0,np.array(list(parametersDefault.defaultInitialConditions.values())).reshape(9,-1)),
                                                stepSize = 0.001,
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



if __name__ == "__main__":
    run()

