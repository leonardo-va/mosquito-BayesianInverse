import json
import argparse
import os
from jsonToModelCode import generate_py_model_function, generate_stan_ode_code, validate_setup
import numpy as np
from matplotlib import pyplot as plt
import runModel
import runSampler
import quantityOfInterest
import sampleEvaluation
import visualization


def _get_root_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    return parent_dir

def _get_setup_path():
    root_dir = _get_root_dir()
    default_setup_path = os.path.join(root_dir, "setup.json")
    return default_setup_path

class cubic_smoothed_function:
    original_function = None
    original_function_derivative = None
    smoothed_function = None
    smooth_intervals = [[0,0]]
    cubic_polys = []


    def __init__(self, function, derivative, intervals:list[list]):
        self.original_function = function
        self.original_function_derivative = derivative
        self.smooth_intervals = intervals
        self._validate_input()

        for interval in self.smooth_intervals:
            lhs_tuple  = self._eval_conditions(interval)
            print(f"lhs tuple (f_0,f_1,f_prime_0,f_prime_1): {lhs_tuple}")
            cubic_poly = fit_cubic_poly(*lhs_tuple, interval[0], interval[1])
            self.cubic_polys.append(cubic_poly)
        self.smoothed_function = self._assemble_function()

    def _validate_input(self):
        for interval in self.smooth_intervals:
            assert(interval[0]<interval[1])
    
    def _test_result(self):
        for idx, interval in enumerate(self.smooth_intervals):
            print(f"interval {idx}: [{interval[0]},{interval[1]}]", flush=True)
            print(f"x_0: {interval[0]} , f(x_0): {self.original_function(interval[0])}, poly(x_0): {self.cubic_polys[idx](interval[0])}",flush=True)
            print(f"x_1: {interval[1]} , f(x_1): {self.original_function(interval[1])}, poly(x_1): {self.cubic_polys[idx](interval[1])}",flush=True)
            # self.original_function(interval[1])

    def _eval_conditions(self,interval):
        f_0 = self.original_function(interval[0])
        f_1 = self.original_function(interval[1])
        f_prime_0 = self.original_function_derivative(interval[0])
        f_prime_1 = self.original_function_derivative(interval[1])
        return (f_0,f_1,f_prime_0,f_prime_1)
    
    def _assemble_function(self):
        def smooth_function(x):
            retval = None
            for idx,interval in enumerate(self.smooth_intervals):
                if(x>=interval[0] and x<= interval[1]):
                    retval = self.cubic_polys[idx](x)
            if(retval is None):
                retval = self.original_function(x)
            return retval
        return smooth_function
    
    def get_smoothed_function(self):
        return self.smoothed_function

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
    if((temperature>max(0,T_min_fit)) and (temperature <T_max_fit)):
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

def cubic_polynomial(x, cubic_coeffs):
    return (np.array([x**3, x**2, x, 1]).dot(cubic_coeffs)).item()

def fit_cubic_poly(f_0, f_prime_0, f_1, f_prime_1, x_0, x_1):
    lhs = np.array([f_0,f_1,f_prime_0,f_prime_1]).reshape((4,1))
    lgs_mat = np.array([[x_0**3, x_0**2, x_0, 1],
                        [x_1**3, x_1**2, x_1, 1],
                        [3*x_0**2,2*x_0, 1, 0],
                        [3*x_1**2,2*x_1, 1, 0]])
    print(lgs_mat, "lgs")
    print(np.linalg.inv(lgs_mat), "lgs_inv")
    cubic_coeffs = np.linalg.inv(lgs_mat) @ lhs
    return lambda x: cubic_polynomial(x, cubic_coeffs)




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
    # print(temperature_profile(np.arcsin(1/2)*100/np.pi))
    # print(temperature_profile((np.pi - np.arcsin(1/2))*100/np.pi))
    times = np.linspace(0,100,500)
    temps = np.array([temperature_profile(t) for t in times])
    # plt.plot(times,temps)
    # plt.show()
    biting_params = np.array([biting_rate_raw(temperature_profile(t)) for t in times])
    smoothing = cubic_smoothed_function(biting_rate_raw, biting_rate_derivative, [[-5,5],[40,50]])
    smoothing._test_result()
    biting_rate_smooth = smoothing.get_smoothed_function()
    print(times)
    print(biting_rate_smooth(44.89795918))
    biting_params_smooth = [biting_rate_smooth(t) for t in times]
    print(biting_params_smooth)
    plt.plot(times,biting_params_smooth)
    plt.show()

    # mosquito_model = generate_py_model_function(setup, generated_py_function_file)
    # stan_code = generate_stan_ode_code(setup, generated_stan_code_file)
    # parameters = setup['parameters']
    # initial_state = setup['initial_state']
    # noise = setup['observable_standard_deviation']

    # # generateDefaultSetup(os.path.join(_get_root_dir(), "defaultsetup.json"))
    
    # try:
    #     runModel.generate_report_plots(mosquito_model, parameters, initial_state, 'RK4', save_png_dir = _get_root_dir())
    #     pass
    # except:
    #     print("generating plots failed, continuing with data generation and sampling")

    # # generate data for the sampler
    # observables = []
    # for observable in setup["state_to_observable"]:
    #     linearCoefficients = observable["linear_combination"]
    #     observables.append(lambda interpolant: quantityOfInterest.linearCombinationQOI(interpolant, linearCoefficients))
    # data = runModel.generate_data_from_setup(mosquito_model,setup)
    # qoi_names = [observable['name'] for observable in setup['state_to_observable']]
    # visualization.visualize_artificial_data(data, qoi_names)
    
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