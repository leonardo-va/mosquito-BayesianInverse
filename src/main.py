import json
import argparse
import os
from jsonToModelCode import generate_py_model_function, generate_stan_ode_code, validate_setup
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
    mosquito_model = generate_py_model_function(setup, generated_py_function_file)
    stan_code = generate_stan_ode_code(setup, generated_stan_code_file)
    parameters = setup['parameters']
    initial_state = setup['initial_state']
    noise = setup['observable_standard_deviation']

    # generateDefaultSetup(os.path.join(_get_root_dir(), "defaultsetup.json"))
    
    try:
        runModel.generate_report_plots(mosquito_model, parameters, initial_state, 'RK4', save_png_dir = _get_root_dir())
    except:
        print("generating plots failed, continuing with data generation and sampling")

    # generate data for the sampler
    observables = []
    for observable in setup["state_to_observable"]:
        linearCoefficients = observable["linear_combination"]
        observables.append(lambda interpolant: quantityOfInterest.linearCombinationQOI(interpolant, linearCoefficients))
    data = runModel.generate_data_from_setup(mosquito_model,setup)
    qoi_names = [observable['name'] for observable in setup['state_to_observable']]
    visualization.visualize_artificial_data(data, qoi_names)
    
    # build and run the sampler 
    samples_dataframe = runSampler.sample(stan_code = stan_code, data = data.noisyData, num_samples = setup["number_of_samples"])

    samples_csv_path = runSampler.save_samples(samples_dataframe, os.path.join(_get_root_dir(), "samples"), setup)

    sampleEvaluation.sampleEvaluation(samples_dataframe, 
                     setup['parameters'], 
                     saveResultPath=f"{os.path.splitext(samples_csv_path)[0]}_evaluation.png")
    sampleEvaluation.compare_data_and_prediction(samples_dataframe, setup)

if __name__ == "__main__":
    main()