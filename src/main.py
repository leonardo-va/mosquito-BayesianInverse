import json
import argparse
import os
from jsonToModelCode import generate_py_model_function, generate_stan_ode_code
import runModel
import runSampler
import quantityOfInterest
import sampleEvaluation


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

    # generate the model function, parameters, and stan code from the setup
    generated_stan_code_file = os.path.join(_get_root_dir(), "stan_code.txt")
    generated_py_function_file = os.path.join(_get_root_dir(), "py_model_function.txt")
    mosquito_model = generate_py_model_function(setup, generated_py_function_file)
    stan_code = generate_stan_ode_code(setup, generated_stan_code_file)
    parameters = setup['parameters']
    initial_state = setup['initial_state']

    # generateDefaultSetup(os.path.join(_get_root_dir(), "defaultsetup.json"))
    # solve the model equation and generate some plots (this is just for visualizing, and is not necessary for sampling)
    runModel.run(mosquito_model, parameters, initial_state, 'RK4', save_png_dir = _get_root_dir())

    # generate data for the sampler
    observables = []
    for observable in setup["state_to_observable"]:
        linearCoefficients = observable["linear_combination"]
        observables.append(lambda interpolant: quantityOfInterest.linearCombinationQOI(interpolant, linearCoefficients))
    data = runModel.generateData(mosquito_model, 
                                 quantitiesOfInterest = observables,
                                 numberObservations = setup["number_of_measurements"],
                                 timeInterval = setup["time_interval"],
                                 parameters = parameters,
                                 initialState = initial_state)
    
    # build and run the sampler 
    samples_dataframe = runSampler.sample(stan_code = stan_code, data = data, num_samples = setup["number_of_samples"])

    samples_csv_path = runSampler.save_samples(samples_dataframe, os.path.join(_get_root_dir(), "samples"), setup)

    f"{os.path.splitext(samples_csv_path)[0]}_evaluation.png"
    sampleEvaluation.sampleEvaluation(samples_dataframe, 
                     generateDataParameters=setup["parameters"], 
                     saveResultPath=f"{os.path.splitext(samples_csv_path)[0]}_evaluation.png")
    sampleEvaluation.compare_observables(samples_dataframe, setup)

if __name__ == "__main__":
    main()