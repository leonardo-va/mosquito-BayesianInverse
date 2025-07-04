from pandas import read_csv, DataFrame
from matplotlib import pyplot as plt
import numpy as np
import json
import argparse
import os
import odeSolver
from jsonToModelCode import generate_py_model_function
import visualization

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
        print(f"samples mean {parameterName}: {samplesMean}")
        currentAx.hist(samples,nBins,alpha=0.6)
        if(generateDataParameters is not None):
            currentAx.axvline(generateDataParameters[parameterName], color='lightgreen', label=f"true value: {generateDataParameters[parameterName]:.4g}")
            currentAx.axvline(samplesMean, color='red', label=f"samples mean: {samplesMean:.4g}")
        currentAx.set_title(parameterName)
        currentAx.legend()
    if(saveResultPath is not None):
        fig.savefig(saveResultPath)
    plt.show()

def sample_evaluation_from_csv(samples_csv_path:str, generate_data_parameters:dict=None):
    samplesDF = read_csv(samples_csv_path)
    save_evaluation_path = f"{os.path.splitext(samples_csv_path)[0]}_evaluation.png"
    sampleEvaluation(samplesDF, generate_data_parameters, save_evaluation_path)

def compare_observables(samplesDF, setup:dict):
    real_parameters = setup["parameters"]
    posterior_prameters = real_parameters.copy()

    for idx, parameterName in enumerate(setup["inferred_parameters"]):
        samples = samplesDF[parameterName]
        samplesMean = np.mean(samples)
        posterior_prameters[parameterName] = samplesMean

    solver = odeSolver.ODESolver()    
    model_function = generate_py_model_function(setup)
    _, solution_interpolant_real = solver.solve(lambda t,u: model_function(t,u,list(real_parameters.values())), 
                                            setup["time_interval"][1], 
                                            (setup["time_interval"][0],list(setup["initial_state"].values())))
    _, solution_interpolant_posterior = solver.solve(lambda t,u: model_function(t,u,list(posterior_prameters.values())),
                                                  setup["time_interval"][1],
                                                  (setup["time_interval"][0],list(setup["initial_state"].values())))
    visualization.plotMosquitoPopulation([solution_interpolant_real, solution_interpolant_posterior])
    visualization.compare_qoi(solution_interpolant_real, solution_interpolant_posterior, setup["state_to_observable"][0]["linear_combination"])


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
    print(setup_json_path)
    if os.path.isfile(setup_json_path):
        with open(setup_json_path, 'r') as setup_file:
            setup = json.load(setup_file)
            generate_data_parameters = setup["parameters"]
            sample_evaluation_from_csv(args.sample_path, generate_data_parameters)
            samplesDF = read_csv(args.sample_path)
            compare_observables(samplesDF, setup)
        return
    else:
        sample_evaluation_from_csv(args.sample_path)
        return


if __name__ == "__main__":
    main()