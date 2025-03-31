from odeSolver import PiecewiseLinearInterpolant
import numpy as np

def linearCombinationQOI(interpolant: PiecewiseLinearInterpolant, coefficients:list)->PiecewiseLinearInterpolant:
    
    if(len(coefficients)!=interpolant.codomainDim()):
        print(f'Warning: linear combination coefficients has length {len(coefficients)}, but interpolant codomain has dimension {interpolant.codomainDim()}')
        return None
    
    QOIValues = np.matmul(interpolant.values,np.array(coefficients))
    QOI_Interpolant = PiecewiseLinearInterpolant(interpolant.grid, QOIValues)
    return QOI_Interpolant

def numberOfHosts(interpolant:PiecewiseLinearInterpolant):
    return linearCombinationQOI(interpolant, [0,0,0,0,0,1,1,1,1])
 
def numberOfMosquitos(interpolant:PiecewiseLinearInterpolant):
    return linearCombinationQOI(interpolant, [0,0,1,1,1,0,0,0,0])

def mosquitoToHostRatio(interpolant: PiecewiseLinearInterpolant):
    ratioValues = numberOfMosquitos(interpolant).values/numberOfHosts(interpolant).values
    return PiecewiseLinearInterpolant(interpolant.grid, ratioValues)