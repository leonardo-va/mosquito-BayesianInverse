import numpy as np

def create_briere_function(q, c, Tmin, Tmax):
    '''
    parameters to fit: q, Tmin, Tmax
    c is a scaling factor such that q/c is on the same scale as tmin and tmax
    return (func,derivative, derivative_discontinuities) tuple
    '''
    def func(T):
        if((T>np.max([0,Tmin])) and (T <Tmax)):
            retval = q/c*T*(T-Tmin)*np.sqrt(Tmax-T)
        else: retval = 0
        return retval
    def derivative(T):
        if((T>max(0,Tmin)) and (T < Tmax)):
            retval =  q/c * ((2*T-Tmin) * np.sqrt(Tmax - T) - T/2*(T-Tmin)/np.sqrt(Tmax-T))
        else:
            retval = 0
        return retval
    derivative_discontinuities = [np.max([Tmin,0]), Tmax]

    return func, derivative, derivative_discontinuities

