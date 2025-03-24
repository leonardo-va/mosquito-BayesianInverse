import numpy as np
from matplotlib import pyplot as plt
import parametersDefault

class PiecewiseLinearInterpolant():
    def __init__(self, grid, values):
        self.grid = grid
        self.values = values
    def eval(self, x):
        if (x<self.grid[0] or x> self.grid[-1]):
            print(f"{x} out of range")
            return None
        right = np.argwhere(self.grid>x)[0][0]
        left = right-1
        dx = (x-self.grid[left])/(self.grid[right]-self.grid[left])
        res = self.values[left] + (self.values[right]-self.values[left])*dx
        return res
       

class ODESolver():
    def solve(self, odeRHS, T, initialCondition, stepSize = 0.01, method="RK4"):
        print(initialCondition)
        if T<initialCondition[0]:
            print(f"{T} is out of range.")
            return None
        NSteps = int(np.ceil(T/stepSize))+1
        currentState = (initialCondition[0], np.array(initialCondition[1]).reshape((1,-1)))
        dimPhaseSpace = int(np.max(currentState[1].shape))
        solutionArray = np.zeros((NSteps+1, dimPhaseSpace))
        grid = np.zeros((NSteps+1,1))
        solutionArray[0] = currentState[1]
        grid[0] = initialCondition[0]
        for i in range(NSteps):
            newState = currentState[1] + stepSize * self.methods[method](self,odeRHS, currentState, stepSize)
            currentState = (currentState[0] + stepSize, newState)
            solutionArray[i+1] = currentState[1]
            grid[i+1] = currentState[0]
        linearInterpolantSolution = PiecewiseLinearInterpolant(grid, solutionArray)
        return linearInterpolantSolution.eval(T), linearInterpolantSolution

    
    def _incrementRK4(self, odeRHS, currentState, stepsize):
        t = currentState[0]
        u = currentState[1]
        h = stepsize
        k1 = odeRHS(t, u)
        k2 = odeRHS(t+h/2, u+h/2 * k1)
        k3 = odeRHS(t+h/2, u+h/2 * k2)
        k4 = odeRHS(t + h, u + h*k3)
        increment = 1/6*k1 + 1/3*k2 + 1/3*k3 + 1/6*k4
        return increment

    def _incrementEuler(self, odeRHS, currentState, stepsize):
        t = currentState[0]
        u = currentState[1]
        h = stepsize
        increment = odeRHS(t,u)
        return increment
    methods = {"RK4":_incrementRK4,
               "Euler":_incrementEuler
               }


        
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

    print(f'hostDerivative: {np.sum(res[5:])} \n host number: {nh}')
    
    return res.T


# print(parametersDefault.initialConditions.values())
print(np.array(list(parametersDefault.initialConditions.values())).reshape(9,-1)
      )

pointSolution, interpolant = ODESolver().solve(odeRHS = lambda t,u: mosquitoModel(t,u,
                                                list(parametersDefault.parameters.values())), 
                                                T=0.1,
                                                initialCondition=(0,np.array(list(parametersDefault.initialConditions.values())).reshape(9,-1)),
                                                stepSize = 0.001,
                                                method="Euler")

# print(pointSolution)
# print(interpolant.eval(2.5))

# def hostPopulation(u_eval):


# plt.subplots()
plt.scatter(interpolant.grid, interpolant.values[:,7])
plt.show()
