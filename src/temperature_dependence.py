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

def _get_root_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    return parent_dir

def _get_setup_path():
    root_dir = _get_root_dir()
    default_setup_path = os.path.join(root_dir, "setup.json")
    return default_setup_path

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

def constant_function(val):
    return lambda t: val

def briere_func(T, q, c, Tmin, Tmax):
    '''
    parameters to fit: q, Tmin, Tmax
    c is a scaling factor such that q/c is on the same scale as tmin and tmax
    T is the argument (temperature)
    '''
    if((T>np.max([0,Tmin])) and (T <Tmax)):
        retval = q/c*T*(T-Tmin)*np.sqrt(Tmax-T)
    else: retval = 0
    return 10*retval


def temperature_profile(time:float):
    time_15_0 = np.arcsin(1/2)*100/np.pi
    time_15_1 = (np.pi - np.arcsin(1/2))*100/np.pi
    
    if(time<0 or time>100): return 15
    return 30*np.sin((time_15_0 + time/100*(time_15_1-time_15_0))*np.pi/100)

def biting_rate_raw(temperature):
    q_fit = 3.71
    c_scaling = 43000
    T_min_fit = 1.93
    T_max_fit = 45.1
    if((temperature>np.max([0,T_min_fit])) and (temperature <T_max_fit)):
        retval = q_fit/c_scaling*temperature*(temperature-T_min_fit)*np.sqrt(T_max_fit-temperature)
    else: retval = 0
    return retval

def biting_rate_derivative(temperature):
    q = 3.71
    c = 43000
    T_min = 1.93
    T_max = 45.1
    t = temperature
    if((t>max(0,T_min)) and (t <T_max)):
        retval =  q/c * ((2*t-T_min) * np.sqrt(T_max - t) - t/2*(t-T_min)/np.sqrt(T_max-t))
    else:
        retval = 0
    return retval

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
 
    times = np.linspace(0,100,500)
    # temps = np.array([temperature_profile(t) for t in times])
    # # plt.plot(times,temps)
    # # plt.show()
    cubicSmoothing = smoothing.CubicSmoothing(biting_rate_raw, biting_rate_derivative, [[-5,5],[40,50]])
    cubicSmoothing.test_result()
    biting_rate_smooth = cubicSmoothing.get_smoothed_function()
    biting_params_smooth = [biting_rate_smooth(temperature_profile(t)) for t in times]
    plt.plot(times,biting_params_smooth)
    plt.show()
    for pname, param_val in setup['parameters'].items():
        print(pname, param_val)
    params_temp_dependent = {pname:constant_function(param_val) for pname, param_val in setup['parameters'].items()}
    # test biting rate
    params_temp_dependent['a'] = lambda t: biting_rate_smooth(temperature_profile(t))
    print(params_temp_dependent['delta_E'](1), params_temp_dependent['beta'](1))
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