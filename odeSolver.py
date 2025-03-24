import numpy as np

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
    def codomainDim(self):
        return self.values.shape[1]
       

class ODESolver():
    def solve(self, odeRHS, T, initialCondition, stepSize = 0.01, method="RK4"):
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
