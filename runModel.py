import parametersDefault
from odeSolver import ODESolver
import numpy as np
import visualization
from matplotlib import pyplot as plt
import quantityOfInterest

def mosquitoModel(t, u, parameters):
    '''
    Forward model:

    ODE descrbing the population dynamics: u'(t) = f(t,u(t),parameters)

    return: u'  
    '''
    
    u = u.T
    # # eggs
    # e = u[0]
    # # juveniles
    # j = u[1]
    # # susceptible mosquitos
    # sm = u[2]
    # # exposed mosquitos
    # em = u[3]
    # # infectious mosquitos
    # im = u[4]
    # # susceptible hosts
    # sh = u[5]
    # # exposed hosts
    # eh = u[6]
    # # infectious hosts
    # ih = u[7]
    # # recovered hosts
    # rh = u[8]

    # # constants:
    # # total mosquito population
    # m = sm + em + im
    # # total host population
    # nh = sh + eh + ih + rh

    # # rename parameters
    # delta_E = parameters[0]
    # mu_E = parameters[1]
    # beta = parameters[2]
    # alpha = parameters[3]
    # delta_J = parameters[4]
    # mu_J = parameters[5]
    # omega = parameters[6]
    # mu_M = parameters[7]
    # a = parameters[8]
    # b_M = parameters[9]
    # alpha_M = parameters[10]
    # Lambda = parameters[11]
    # b_H = parameters[12]
    # mu_H = parameters[13]
    # alpha_H = parameters[14]
    # gamma_H = parameters[15]
    
    # # equation
    # res = np.zeros(u.shape)
    # res[0] = beta*m - (delta_E+mu_E)*e
    # res[1] = delta_E*e - alpha*(j**2) - mu_J*j - delta_J*j
    # res[2] = omega*delta_J*j - a*b_M*sm*ih/nh - mu_M*sm
    # res[3] = a*b_M*sm*ih/nh - (alpha_M+mu_M)*em
    # res[4] = alpha_M*em - mu_M*im
    # res[5] = Lambda - a*b_H*im*sh/nh - mu_H*sh
    # res[6] = a*b_H*im*sh/nh - alpha_H*eh - mu_H*eh
    # res[7] = alpha_H*eh-gamma_H*ih-mu_H*ih
    # res[8] = gamma_H*ih - mu_H*rh
    
    res = np.zeros(u.shape)
    res[0] = parameters[2]*(u[2]+u[3]+u[4]) - (parameters[0]+parameters[1])*u[0]
    res[1] = parameters[0]*u[0] - parameters[3]*(u[1]**2) - parameters[5]*u[1] - parameters[4]*u[1]
    res[2] = parameters[6]*parameters[4]*u[1] - parameters[8]*parameters[9]*u[2]*u[7]/(u[5]+u[6]+u[7]+u[8]) - parameters[7]*u[2]
    res[3] = parameters[8]*parameters[9]*u[2]*u[7]/(u[5]+u[6]+u[7]+u[8]) - (parameters[10]+parameters[7])*u[3]
    res[4] = parameters[10]*u[3] - parameters[7]*u[4]
    res[5] = parameters[11] - parameters[8]*parameters[12]*u[4]*u[5]/(u[5]+u[6]+u[7]+u[8]) - parameters[13]*u[5]
    res[6] = parameters[8]*parameters[12]*u[4]*u[5]/(u[5]+u[6]+u[7]+u[8]) - parameters[14]*u[6] - parameters[13]*u[6]
    res[7] = parameters[14]*u[6]-parameters[15]*u[7]-parameters[13]*u[7]
    res[8] = parameters[15]*u[7] - parameters[13]*u[8]
    return res.T

def generateData(N:int, timeInterval: list, parameters: dict, initialCondition: dict, quantitiesOfInterest : list, solverMethod = 'RK4')->dict:
    
    initial = (timeInterval[0], list(initialCondition.values()))
    solver = ODESolver()
    _, interpolantMosquito = solver.solve(lambda t,u: mosquitoModel(t,u,list(parameters.values())), timeInterval[1], initial, 0.001, solverMethod)
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
        _, solutionInterpolant = ODESolver().solve(odeRHS = lambda t,u: mosquitoModel(t,u,
                                                    list(params.values())), 
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
    _, solutionInterpolant = ODESolver().solve(odeRHS = lambda t,u: mosquitoModel(t,u,
                                                list(parametersDefault.defaultParameters.values())), 
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

run()



