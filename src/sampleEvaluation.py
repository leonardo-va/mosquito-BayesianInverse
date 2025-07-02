from pandas import read_csv, DataFrame
from matplotlib import pyplot as plt
import numpy as np
import json
import argparse
import os

def sampleEvaluation(samplesDF:DataFrame, generateDataParameters:dict = None):
    inferedParameterNames = []
    for colname in samplesDF.columns.tolist():
        if (not colname.endswith("__") and not colname=='draws'):
            inferedParameterNames.append(colname)
    nSubplotRows = int(np.ceil(len(inferedParameterNames)/3))
    nBins = 50
    fig, axs = plt.subplots(nSubplotRows,3)
    if(nSubplotRows == 1):
        axs = np.expand_dims(axs, axis=0)
    for idx, parameterName in enumerate(inferedParameterNames):
        fig_row, fig_col = int(np.floor(idx/3)),idx%3
        currentAx = axs[fig_row, fig_col]
        samples = samplesDF[parameterName]
        print(f"samples mean {parameterName}: {np.mean(samples)}")
        currentAx.hist(samples,nBins)
        if(generateDataParameters is not None):
            currentAx.axvline(generateDataParameters[parameterName], color='red')
        currentAx.set_title(parameterName)
    plt.show()

def sample_evaluation_from_csv(samples_csv_path:str, generate_data_parameters:dict=None):
    samplesDF = read_csv(samples_csv_path)
    sampleEvaluation(samplesDF, generate_data_parameters)

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
        return
    else:
        sample_evaluation_from_csv(args.sample_path)
        return


if __name__ == "__main__":
    main()