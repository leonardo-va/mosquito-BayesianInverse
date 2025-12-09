import os
import argparse
import json
from jsonToModelCode import validate_setup, generate_py_model_function
import temperature_dependent_parameters as tdp
import quantityOfInterest
import runModel
import visualization
import temp_dependent_stan_code
import py_model_creator
import pymc as pm
import pytensor.tensor as pt
import pytensor.printing as ptp
import numpy as np
import pytensor

def _get_root_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    return parent_dir

def _get_setup_path():
    root_dir = _get_root_dir()
    default_setup_path = os.path.join(root_dir, "setup.json")
    return default_setup_path

def make_func_param_dict(temp_dependent_param_list:list[tdp.tempDependentParameter]):
    fct_param_dict = {}
    for param in temp_dependent_param_list:
        fct_param_dict[param.name] = param.get_function_parameters()
    return fct_param_dict

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
 
    print("Creating ODE-model")
    params_temp_dependent, params_temp_dependent_pt = tdp.make_gt_temp_dependent_parameters(setup)
    model_creator = py_model_creator.PyModelCreator(list(params_temp_dependent.values()), tdp.temperature_profile)
    model_creator_pt = py_model_creator.PyModelCreator(list(params_temp_dependent_pt.values()), tdp.temperature_profile_pt)
    mosquito_model = model_creator.get_model_function()
    mosquito_model_pt = model_creator_pt.get_model_function(pytensor_version=True)
    # print(temp_dependent_stan_code.assemble_function_block(setup, param_metas))
    # print(temp_dependent_stan_code.assemble_data_block(setup, param_metas))
    # print(temp_dependent_stan_code.assemble_parameter_block(setup, param_metas))
    # stan_code = generate_stan_ode_code(setup, generated_stan_code_file)

    initial_state = setup['initial_state']
    sigma_obs_list = setup['observable_standard_deviation']

    # generate data for the sampler
    print("Generating artificial data")
    observables = []
    qoi_names = []
    qoi_mat_cols = []
    for observable in setup["state_to_observable"]:
        linearCoefficients = observable["linear_combination"]
        qoi_mat_cols.append(linearCoefficients)
        qoi_names.append(observable['name'])
        observables.append(lambda interpolant: quantityOfInterest.linearCombinationQOI(interpolant, linearCoefficients))
    qoi_mat = np.array(qoi_mat_cols).T
    qoi_pt_matrix = pt.constant(qoi_mat)
    setup['parameters'] = make_func_param_dict(params_temp_dependent.values())
    # data = runModel.generate_mock_data(setup)
    data = runModel.generate_data_from_setup(mosquito_model,setup)
    visualization.visualize_artificial_data(data, qoi_names)
    print(data.noisyData)
    print("Building pymc ode")

    theta_li=[]
    for param in params_temp_dependent.values():
        theta_li.append(param.get_function_parameters())
    print(theta_li)
    # theta_ve = pt.concatenate([pt.reshape(x, (-1,)) for x in theta_li])
    # ode_solution = ode_model(y0=pt.as_tensor_variable(list(initial_state.values())), theta=theta_ve)
    print("Building pymc model")
    def pt_debugger(item, msg = "DEBUGMSG"):
        debug = ptp.Print(msg)(item)
        f = pytensor.function([], debug)
        f()   
    obs_noise = setup['observable_standard_deviation']
    print("---------------------------")
    
    with pm.Model() as model:
        # Priors for parameters
        priors = {}
        theta_list = []
        # for inferred_param in setup['inferred_parameters']:
        #     param_len = params_temp_dependent[inferred_param].get_function_parameter_length()
        #     mu = pt.constant(params_temp_dependent[inferred_param].get_function_parameters()) 
        #     if(inferred_param == "alpha"):
        #         cov = pt.ones(param_len)*0.001
        #     else:                    
        #         cov = pt.ones(param_len) * 1.0               
        #     priors[inferred_param] = pm.Normal(inferred_param,mu, cov,shape=param_len)
        mu = [3,3,50]
        sigma = [1,1,5]
        priors = {"delta_E": pm.Normal("delta_E", mu=mu, sigma = sigma, size=len(mu))}


        ode_model = pm.ode.DifferentialEquation(
            func=mosquito_model_pt,
            times=data.noisyData['ts'],
            n_states=len(initial_state),
            n_theta=model_creator.get_parameter_length(),
            t0=data.noisyData['t0'])
        for param in params_temp_dependent.values():
            if param.name =="delta_E":
                theta_list.append(priors[param.name])
            else:
                theta_list.append(pt.constant(param.get_function_parameters()))
     
        theta_vector = pt.concatenate([pt.reshape(x, (-1,)) for x in theta_list])
        # single_eval = mosquito_model_pt(pt.as_tensor_variable(list(initial_state.values())),1,theta_vector)

        # ODE solution at observation times
        ode_solution = ode_model(y0=pt.as_tensor_variable(list(initial_state.values())), theta=theta_vector)
        ode_solution_transformed = ode_solution @ qoi_pt_matrix
        pt_debugger(ode_solution_transformed)
        #Likelihood
        for i in range(len(observables)):
            pm.Normal(
                f"y_hat_{i}",
                mu=ode_solution_transformed[:, i],
                sigma=obs_noise[i],
                observed=np.array(data.noisyData['y'])[:, i]
            )
        trace = pm.sample(1000, tune=1000, cores=2, step=pm.Metropolis())
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