import json
import argparse
import inspect
import os
from jsonToModelCode import validate_setup, generate_py_model_function
import numpy as np
from matplotlib import pyplot as plt
import smoothing
import runModel
from functools import wraps
import quantityOfInterest
# import runSampler
# import sampleEvaluation
import visualization
import math

def _get_root_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    return parent_dir

def _get_setup_path():
    root_dir = _get_root_dir()
    default_setup_path = os.path.join(root_dir, "setup.json")
    return default_setup_path

def create_constant_function(val):
    '''
    return (func,derivative, derivative_discontinuities)
    '''
    return (lambda t: val, lambda t: 0, [])

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

def quadratic_zeros(a,b,c):
    if(math.isclose(a,0)):
        return None
    r_term = b**2-4*a*c
    if(math.isclose(r_term, 0)):
        r_term = 0
    if(r_term<0):
        return None
    sol_1 = (-b+r_term)/2*a
    sol_2 = (-b-r_term)/2*a
    return sol_1, sol_2

def create_quadratic_function(q, c, Tmin, Tmax):
    def quadratic_expression(T):
        return q/c*(T-Tmin)*(Tmax-T)
    q_zeros = quadratic_zeros(-q/c,q/c*(Tmin+Tmax),-(q/c*Tmin*Tmax+1))
    derivative_discontinuities = [Tmin, Tmax]
    if(q_zeros is not None):
        for q_zero in q_zeros:
            if(Tmin < q_zero < Tmax):derivative_discontinuities.append(q_zeros[0])
    def func(T):
        retval = 0
        if(T>Tmin and T<Tmax):
            retval = min(quadratic_expression(T),1)
        return retval
    def derivative(T):
        retval = 0
        if(T>Tmin and T<Tmax):
            if(quadratic_expression(T)<=1):
                retval = q/c*(Tmax+Tmin-2*T)
        return retval
    return func, derivative, derivative_discontinuities

def create_truncated_linear(beta, Tmax):
    #Tmax = alpha/beta
    def func(T):
        retval = 0
        if(T<Tmax):
            retval = -beta*(T-Tmax)
        return retval
    def derivative(T):
        retval = 0
        if(T<Tmax):
            retval = -beta
        return retval
    derivative_discontinuities = [Tmax]
    return func, derivative, derivative_discontinuities

def add_temperature_dependence(func, parameters_temp_dependent):
    sig = inspect.signature(func)
    @wraps(func)
    def func_temp_dependent(*args, **kwargs):
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        if 'parameters' and 't' in bound_args.arguments:
            current_t = bound_args.arguments['t']
            bound_args.arguments['parameters'] = [temp_func(current_t) for _,temp_func in parameters_temp_dependent.items()]
        if 'setup' in bound_args.arguments:
            bound_args.arguments['setup']['parameters'] = parameters_temp_dependent
        return func(*bound_args.args, **bound_args.kwargs)
    return func_temp_dependent

def inject_temperature_profile(func):
    def func_with_temp_profile(t):
        return func(temperature_profile(t))
    return func_with_temp_profile

class tempDependentParameterMeta:
    name:str
    long_name:str
    function_type:str
    function_parameters:str
    # smoothing_intervals:list
    def __init__(self, name = "", function_type="", function_parameters=[],long_name = ""):
        self.name = name
        self.long_name = long_name
        self.function_type = function_type
        self.function_parameters = function_parameters
        # self.smoothing_intervals = smoothing_intervals

class tempDependentParameter:
    name:str
    long_name:str
    function_type:str
    function_parameters:list
    func:callable
    function_type_list = {
        "briere":create_briere_function,
        "quadratic":create_quadratic_function,
        "constant":create_constant_function,
        "truncated_linear":create_truncated_linear
        }
    def __init__(self, name = "", func = None, function_type="", function_parameters=[], long_name = ""):
        self.name = name
        self.long_name = long_name
        self.func = func
        self.function_type = function_type
        self.function_parameters = function_parameters
    
    @classmethod
    def from_metadata(cls, meta:tempDependentParameterMeta):
        '''
        this smoothes the function
        '''
        if(meta.function_type not in cls.function_type_list):
            raise ValueError(f"function type \"{meta.function_type}\" does not exist")
        func_creator = cls.function_type_list[meta.function_type]
        func, derivative, derivative_discontinuities = func_creator(*meta.function_parameters)
        # if(len(derivative_discontinuities)>0):
        #     smoothing_intervals = smoothing.estimate_smoothing_intervals(derivative_discontinuities)
        #     smoother = smoothing.CubicSmoothing(func, derivative, smoothing_intervals)
        #     func = smoother.get_smoothed_function()
        init_args = [meta.name, func, meta.function_type, meta.function_parameters, meta.long_name]
        return cls(*init_args)
    
    def visualize(self, time_dependent = False):
        if(time_dependent):
            xaxis = np.linspace(0,100,500)
            xlabel = "time(days)"
        else:
            xaxis = np.linspace(-10,60,500)
            xlabel = "temp(degrees C)"

        if(time_dependent):
            evals = [self.func(temperature_profile(t)) for t in xaxis]
        else:
            evals = [self.func(t) for t in xaxis]
        plt.plot(xaxis,evals)
        plt.title(f"{self.name} / {self.long_name}")
        plt.xlabel(xlabel)
        plt.ylabel(self.name)
        plt.show()


def temperature_profile(time:float):
    time_15_0 = np.arcsin(1/2)*100/np.pi
    time_15_1 = (np.pi - np.arcsin(1/2))*100/np.pi
    
    if(time<0 or time>100): return 15
    return 30*np.sin((time_15_0 + time/100*(time_15_1-time_15_0))*np.pi/100)

def make_gt_temp_dependent_parameters(setup:dict)->dict[str,tempDependentParameter]:
    #some are temporary treated as constants. update this if adding new temp dependent ones
    # _params_in_order = ["delta_E", "p_E", "beta", "alpha", "delta_J", "p_J", "omega", "lf_M", "a", "b_M", "alpha_M", "Lambda", "b_H", "mu_H", "alpha_H", "gamma_H"]
    _params_in_order = setup["parameters"].keys()
    _temp_dependent_indices = [0,1,4,5,7,8]
    # _constant_indices = []
    # for idx,_ in _params_in_order:
    #     if idx not in _temp_dependent_indices:
    #         _constant_indices.append(idx)
    temp_dependent_param_metas = [
    tempDependentParameterMeta("delta_E", "briere", [3.43, 1.6*10**4, 2.53, 51.69], "egg hatching rate"),
    # scaling c is not known for this. Use cx.pallens data (cx.pipiens missing)
    tempDependentParameterMeta("p_E", "quadratic", [5.49, 1.6*10**4, 6.56, 39.52], "egg viability"),
    tempDependentParameterMeta("delta_J", "briere", [3.96, 1.6*10**4, 1.57, 41.9], "juvenile development rate"),
    tempDependentParameterMeta("p_J", "quadratic", [4.41, 10**3, 7.38, 36.14], "juvenile survival"),
    tempDependentParameterMeta("lf_M", "truncated_linear", [4.42, 34.05], "adult life span"),
    tempDependentParameterMeta("a", "briere", [3.71, 43000, 1.93, 45.1], "biting rate")
    ]
    param_meta_dict = {}
    rel_idx = 0
    for idx, pname in enumerate(_params_in_order):
        if(idx in _temp_dependent_indices):
            param_meta_dict[pname] = temp_dependent_param_metas[rel_idx]
            rel_idx += 1
        else:
            param_meta_dict[pname] = tempDependentParameterMeta(pname, "constant", [setup["parameters"][pname]], "")
    
    param_dict = {pname:tempDependentParameter.from_metadata(pmeta) for pname,pmeta in param_meta_dict.items()}
    return param_dict

def main():
    # take the path to setup.json as a commandline argument, also give setup.json in the parent directory as default argument
    setup_path_default_arg = _get_setup_path()
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", default = setup_path_default_arg, type=str, help="Path to the setup.json file")
    args = parser.parse_args()

    with open(args.setup, 'r') as setup_file:
        setup = json.load(setup_file)

    if validate_setup(setup) == False:
        print("setup is not valid")
        return None
    
    # generate the model function, parameters, and stan code from the setup
    generated_stan_code_file = os.path.join(_get_root_dir(), "stan_code.txt")
    generated_py_function_file = os.path.join(_get_root_dir(), "py_model_function.txt")
 
  
    params_temp_dependent_objs = make_gt_temp_dependent_parameters(setup)
    params_temp_dependent = {}
    for obj in params_temp_dependent_objs.values():
        obj.visualize(time_dependent = True)
        f_concatenated = inject_temperature_profile(obj.func)
        print(f_concatenated(50))
        params_temp_dependent[obj.name] = f_concatenated
    for name,func in params_temp_dependent.items():
        print(f"pname {name} val {func(50)}")
    # params_temp_dependent['a'] = lambda t: biting_rate_smooth(temperature_profile(t))
    # print(params_temp_dependent['a'](20))
    mosquito_model = generate_py_model_function(setup, generated_py_function_file)
    mosquito_model = add_temperature_dependence(mosquito_model, params_temp_dependent)
    generate_data_from_setup = add_temperature_dependence(runModel.generate_data_from_setup, params_temp_dependent)
    # stan_code = generate_stan_ode_code(setup, generated_stan_code_file)
    # parameters = setup['parameters']
    initial_state = setup['initial_state']
    noise = setup['observable_standard_deviation']

    # # generateDefaultSetup(os.path.join(_get_root_dir(), "defaultsetup.json"))
    # try:
    #     runModel.generate_report_plots(mosquito_model, parameters, initial_state, 'RK4', save_png_dir = _get_root_dir())
    #     pass
    # except:
    #     print("generating plots failed, continuing with data generation and sampling")

    # generate data for the sampler
    observables = []
    for observable in setup["state_to_observable"]:
        linearCoefficients = observable["linear_combination"]
        observables.append(lambda interpolant: quantityOfInterest.linearCombinationQOI(interpolant, linearCoefficients))
    data = generate_data_from_setup(mosquito_model,setup)
    qoi_names = [observable['name'] for observable in setup['state_to_observable']]
    visualization.visualize_artificial_data(data, qoi_names)
    
    # # build and run the sampler 
    # samples_dataframe, summary = runSampler.sample(stan_code = stan_code, data = data.noisyData, num_samples = setup["number_of_samples"])


    # samples_csv_path = runSampler.save_samples(samples_dataframe, os.path.join(_get_root_dir(), "samples"), setup)
    # # summary_path =f"{os.path.splitext(samples_csv_path)[0]}_summary.json"
    # sampleEvaluation.show_summary(summary)


    # sampleEvaluation.sampleEvaluation(samples_dataframe, 
    #                  setup['parameters'], 
    #                  saveResultPath=f"{os.path.splitext(samples_csv_path)[0]}_evaluation.png")
    # sampleEvaluation.compare_data_and_prediction(samples_dataframe, setup, save_result_prefix=os.path.splitext(samples_csv_path)[0])

if __name__ == "__main__":
    main()