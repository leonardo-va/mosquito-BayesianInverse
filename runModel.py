import parametersDefault
from odeSolver import *
import numpy as np
from quantityOfInterest import numberOfHosts
from matplotlib import pyplot as plt

def mosquitoModel(t, u, parameters):
    '''
    Forward model:

    ODE descrbing the population dynamics: u'(t) = f(t,u(t),parameters)

    return: u'  
    '''
    
    u = u.T
    # eggs
    e = u[0]
    # juveniles
    j = u[1]
    # susceptible mosquitos
    sm = u[2]
    # exposed mosquitos
    em = u[3]
    # infectious mosquitos
    im = u[4]
    # susceptible hosts
    sh = u[5]
    # exposed hosts
    eh = u[6]
    # infectious hosts
    ih = u[7]
    # recovered hosts
    rh = u[8]

    # constants:
    # total mosquito population
    m = sm + em + im
    # total host population
    nh = sh + eh + ih + rh

    # rename parameters
    delta_E = parameters[0]
    mu_E = parameters[1]
    beta = parameters[2]
    alpha = parameters[3]
    delta_J = parameters[4]
    mu_J = parameters[5]
    omega = parameters[6]
    mu_M = parameters[7]
    a = parameters[8]
    b_M = parameters[9]
    alpha_M = parameters[10]
    Lambda = parameters[11]
    b_H = parameters[12]
    mu_H = parameters[13]
    alpha_H = parameters[14]
    gamma_H = parameters[15]
    
    # print("h pop:", m)
    # equation
    res = np.zeros(u.shape)
    res[0] = beta*m - (delta_E+mu_E)*e
    res[1] = delta_E*e - alpha*(j**2) - mu_J*j - delta_J*j
    res[2] = omega*delta_J*j - a*b_M*sm*ih/nh - mu_M*sm
    res[3] = a*b_M*sm*ih/nh - (alpha_M+mu_M)*em
    res[4] = alpha_M*em - mu_M*im
    res[5] = Lambda - a*b_H*im*sh/nh - mu_H*sh
    res[6] = a*b_H*im*sh/nh - alpha_H*eh - mu_H*eh
    res[7] = alpha_H*eh-gamma_H*ih-mu_H*ih
    res[8] = gamma_H*ih - mu_H*rh

    # print(f'hostDerivative: {np.sum(res[5:])} \n host number: {nh}')
    
    return res.T


# print(parametersDefault.initialConditions.values())
# print(np.array(list(parametersDefault.initialConditions.values())).reshape(9,-1)
#       )

pointSolution, solutionInterpolant = ODESolver().solve(odeRHS = lambda t,u: mosquitoModel(t,u,
                                                list(parametersDefault.parameters.values())), 
                                                T=4,
                                                initialCondition=(0,np.array(list(parametersDefault.initialConditions.values())).reshape(9,-1)),
                                                stepSize = 0.001,
                                                method="Euler")

# print(pointSolution)
# print(interpolant.eval(2.5))

# def hostPopulation(u_eval):


nHostsInterpolant = numberOfHosts(solutionInterpolant)
# plt.scatter(solutionInterpolant.grid, solutionInterpolant.values[:,5])
print(nHostsInterpolant.values)
plt.scatter(nHostsInterpolant.grid, nHostsInterpolant.values)
# plt.scatter(solutionInterpolant.grid, solutionInterpolant.values[:,7])
plt.show()
