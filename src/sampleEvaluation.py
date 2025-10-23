from pandas import read_csv, DataFrame
from matplotlib import pyplot as plt
import numpy as np
import json
import argparse
import os
import odeSolver
from jsonToModelCode import generate_py_model_function
import visualization
import quantityOfInterest
from scipy.stats import gaussian_kde

def map_estimate_continuous(values):
    kde = gaussian_kde(values)
    xs = np.linspace(min(values), max(values), 1000)
    ys = kde(xs)
    return xs[np.argmax(ys)]

def show_summary(summary:dict):
    print("----------------------------------\nsummary\n----------------------------------\n")
    num_chains, num_samples, ess_bulk, ess_tail, sampling_time_s, num_warmup = summary.values()
    total_num_samples = num_chains * num_samples
    print(f"time to sample from {num_chains} chains with {num_samples} samples each: {sampling_time_s:.4g} s")
    print(f"average time per sample: {sampling_time_s/(num_chains*num_samples):.4g}s")
    print(f"number of warmup samples per chain: {num_warmup}")
    for param_name,  ess_bulk_val in ess_bulk.data_vars.items():
        print(f"effective sample size (bulk) for {param_name}: {ess_bulk_val.values:.4g} out of {total_num_samples} samples ({100*ess_bulk_val.values/total_num_samples:.4g}%)")
        print(f"effective sample size (tails) for {param_name}: {ess_tail.data_vars[param_name].values:.4g} out of {total_num_samples} samples ({100*ess_tail.data_vars[param_name].values/total_num_samples:.4g}%)")
        print(f"average time per independent sample({param_name}): {sampling_time_s/(num_chains*ess_bulk_val.values):.4g} s")
    print("----------------------------------\n----------------------------------\n")

def sampleEvaluation(samplesDF:DataFrame, generateDataParameters:dict = None, saveResultPath = None):
    inferedParameterNames = []
    for colname in samplesDF.columns.tolist():
        if (not colname.endswith("__") and not colname=='draws'):
            inferedParameterNames.append(colname)
    nSubplotRows = int(np.ceil(len(inferedParameterNames)/3))
    nSamples = samplesDF.shape[0]
    nBins = int(np.ceil(nSamples/100))
    fig, axs = plt.subplots(nSubplotRows,3,figsize=(16,11))
    if(nSubplotRows == 1):
        axs = np.expand_dims(axs, axis=0)
    for idx, parameterName in enumerate(inferedParameterNames):
        fig_row, fig_col = int(np.floor(idx/3)),idx%3
        currentAx = axs[fig_row, fig_col]
        samples = samplesDF[parameterName]
        samplesMean = np.mean(samples)
        samplesMAP = map_estimate_continuous(samples)
        print(f"samples mean {parameterName}: {samplesMean}")
        print(f"samples MAP {parameterName}: {samplesMAP}")
        currentAx.hist(samples,nBins,alpha=0.6)
        if(generateDataParameters is not None):
            currentAx.axvline(generateDataParameters[parameterName], color='lightgreen', label=f"true value: {generateDataParameters[parameterName]:.4g}")
            currentAx.axvline(samplesMean, color='red', label=f"samples mean: {samplesMean:.4g}")
            currentAx.axvline(samplesMAP, color = 'blue', label=f"MAP point: {samplesMAP:.4g}")
        currentAx.set_title(parameterName)
        currentAx.legend()
    if(saveResultPath is not None):
        fig.savefig(saveResultPath)
    plt.show()

# def sampleEvaluation(samplesDF:DataFrame, setup:dict = None, saveResultPrefix = None):
#     inferedParameterNames = []
#     prior_list = {"normal": np.random.normal,
#                   "lognormal": np.random.lognormal,
#                   "uniform":np.random.uniform}
#     for colname in samplesDF.columns.tolist():
#         if (not colname.endswith("__") and not colname=='draws'):
#             inferedParameterNames.append(colname)
#     for idx, parameterName in enumerate(inferedParameterNames):
#         posterior_samples = samplesDF[parameterName]
#         true_param_value = setup['parameters'][parameterName]
#         if(saveResultPrefix is not None):
#             saveResultPath = saveResultPrefix + f"_{parameterName}.png"
#         # try to sample from prior. The numpy names and stan names for the distribution might be different, then this wont work because the 
#         # setup uses stan names
#         try:
#             distribution = setup['inferred_parameters'][parameterName]['distribution']
#             distribution_parameters = setup['inferred_parameters'][parameterName]['parameters']
#             prior_samples = prior_list[distribution](*distribution_parameters, setup["number_of_samples"])
#         except:
#             prior_samples = None
#             print("sample evaluation :: failed prior sampling, visualize without prior")
#         visualization.visualize_prior_to_posterior(posterior_samples, 
#                                                    parameterName, 
#                                                    true_param_value, 
#                                                    prior_samples = prior_samples, 
#                                                    save_result_path = saveResultPath)
        


def compare_data_and_prediction(samplesDF, setup:dict, save_result_prefix):
    '''
    Compare the observables the true parameters would generate, with the observables the posterior mean would
    predict.
    '''
    real_parameters = setup["parameters"]
    posterior_mean = real_parameters.copy()
    posterior_map = real_parameters.copy()

    for idx, parameterName in enumerate(setup["inferred_parameters"]):
        samples = samplesDF[parameterName]
        samplesMean = np.mean(samples)
        samplesMAP = map_estimate_continuous(samples)
        posterior_mean[parameterName] = samplesMean
        posterior_map[parameterName] = samplesMAP

    solver = odeSolver.ODESolver()    
    model_function = generate_py_model_function(setup)
    _, solution_interpolant_real = solver.solve(lambda t,u: model_function(t,u,list(real_parameters.values())), 
                                            setup["time_interval"][1], 
                                            (setup["time_interval"][0],list(setup["initial_state"].values())))
    _, solution_interpolant_posterior_mean = solver.solve(lambda t,u: model_function(t,u,list(posterior_mean.values())),
                                                  setup["time_interval"][1],
                                                  (setup["time_interval"][0],list(setup["initial_state"].values())))
    _, solution_interpolant_posterior_map = solver.solve(lambda t,u: model_function(t,u,list(posterior_map.values())),
                                                  setup["time_interval"][1],
                                                  (setup["time_interval"][0],list(setup["initial_state"].values())))
    for idx, observable in enumerate(setup["state_to_observable"]):
    
        visualization.compare_gt_and_prediction(solution_interpolant_real, solution_interpolant_posterior_mean, 
                                observable,
                                stddev = setup['observable_standard_deviation'][idx],
                                prediction_label = 'mean',
                                save_path = f"{save_result_prefix}_{observable['name']}_mean_prediction.png")
        visualization.compare_gt_and_prediction(solution_interpolant_real, solution_interpolant_posterior_map, 
                                observable,
                                stddev = setup['observable_standard_deviation'][idx],
                                prediction_label = 'MAP',
                                save_path = f"{save_result_prefix}_{observable['name']}_map_prediction.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_path", type=str, help="Path to the samples file")
    args = parser.parse_args()

    if not os.path.isfile(args.sample_path):
        print(f"Error: '{args.sample_path}' is not a valid file.")
        return
    
    print(f"Evaluating samples: {args.sample_path}")
    sample_path_no_extension = os.path.splitext(args.sample_path)[0]
    setup_json_path = f"{sample_path_no_extension}_setup.json"
    save_evaluation_path = f"{sample_path_no_extension}_evaluation.png"
    samplesDF = read_csv(args.sample_path)
    if os.path.isfile(setup_json_path):
        with open(setup_json_path, 'r') as setup_file:
            setup = json.load(setup_file)
            generate_data_parameters = setup["parameters"]
            # sample_evaluation_from_csv(args.sample_path, generate_data_parameters)
            sampleEvaluation(samplesDF, generate_data_parameters, saveResultPath=save_evaluation_path)
            compare_data_and_prediction(samplesDF, setup, sample_path_no_extension)
        return
    else:
        sampleEvaluation(samplesDF, saveResultPath=save_evaluation_path)
        return


if __name__ == "__main__":
    main()