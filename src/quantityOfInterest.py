from odeSolver import PiecewiseLinearInterpolant
import numpy as np

def linearCombinationQOI(interpolant: PiecewiseLinearInterpolant, coefficients:list)->PiecewiseLinearInterpolant:
    
    if(len(coefficients)!=interpolant.codomainDim()):
        print(f'Warning: linear combination coefficients has length {len(coefficients)}, but interpolant codomain has dimension {interpolant.codomainDim()}')
        return None
    
    QOIValues = np.matmul(interpolant.values,np.array(coefficients))
    QOI_Interpolant = PiecewiseLinearInterpolant(interpolant.grid, QOIValues)
    return QOI_Interpolant

def eggs(interpolant: PiecewiseLinearInterpolant):
    return linearCombinationQOI(interpolant, [1,0,0,0,0,0,0,0,0])
def juveniles(interpolant: PiecewiseLinearInterpolant):
    return linearCombinationQOI(interpolant, [0,1,0,0,0,0,0,0,0])
def S_Mosquitoes(interpolant: PiecewiseLinearInterpolant):
    return linearCombinationQOI(interpolant, [0,0,1,0,0,0,0,0,0])
def E_Mosquitoes(interpolant: PiecewiseLinearInterpolant):
    return linearCombinationQOI(interpolant, [0,0,0,1,0,0,0,0,0])
def I_Mosquitoes(interpolant: PiecewiseLinearInterpolant):
    return linearCombinationQOI(interpolant, [0,0,0,0,1,0,0,0,0])
def S_Hosts(interpolant: PiecewiseLinearInterpolant):
    return linearCombinationQOI(interpolant, [0,0,0,0,0,1,0,0,0])
def E_Hosts(interpolant: PiecewiseLinearInterpolant):
    return linearCombinationQOI(interpolant, [0,0,0,0,0,0,1,0,0])
def I_Hosts(interpolant: PiecewiseLinearInterpolant):
    return linearCombinationQOI(interpolant, [0,0,0,0,0,0,0,1,0])
def R_Hosts(interpolant: PiecewiseLinearInterpolant):
    return linearCombinationQOI(interpolant, [0,0,0,0,0,0,0,0,1])
def fullSolution(interpolant: PiecewiseLinearInterpolant):
    return interpolant

def numberOfHosts(interpolant:PiecewiseLinearInterpolant):
    return linearCombinationQOI(interpolant, [0,0,0,0,0,1,1,1,1])
 
def numberOfMosquitos(interpolant:PiecewiseLinearInterpolant):
    return linearCombinationQOI(interpolant, [0,0,1,1,1,0,0,0,0])

def numberOfEggs(interpolant:PiecewiseLinearInterpolant):
    return linearCombinationQOI(interpolant, [1,0,0,0,0,0,0,0,0])

def mosquitoToHostRatio(interpolant: PiecewiseLinearInterpolant):
    ratioValues = numberOfMosquitos(interpolant).values/numberOfHosts(interpolant).values
    return PiecewiseLinearInterpolant(interpolant.grid, ratioValues)

def combine(interpolant1: PiecewiseLinearInterpolant, interpolant2: PiecewiseLinearInterpolant):
    assert(np.all(interpolant1.grid == interpolant2.grid))
    if(len(interpolant1.values.shape) == 1):
        values1 = np.expand_dims(interpolant1.values, axis=1)
    else: values1 = interpolant1.values
    if(len(interpolant2.values.shape) == 1):
        values2 = np.expand_dims(interpolant2.values, axis=1)
    else: values2 = interpolant2.values

    combinedValues = np.concatenate((values1, values2), axis=1)
    combinedInterpolant = PiecewiseLinearInterpolant(interpolant1.grid, combinedValues)
    return combinedInterpolant