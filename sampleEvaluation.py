from pandas import read_csv
from matplotlib import pyplot as plt
import numpy as np
import json

def sampleEvaluation(samplesCSVPath:str, inferedParameterNames:list[str] , generateDataParameters:dict):
    samplesDF = read_csv(samplesCSVPath)
    nSubplotRows = int(np.ceil(len(inferedParameterNames)/3))
    nBins = 50
    print(inferedParameterNames)
    fig, axs = plt.subplots(nSubplotRows,3)
    if(nSubplotRows == 1):
        axs = np.expand_dims(axs, axis=0)
    print(axs.shape)
    for idx, parameterName in enumerate(inferedParameterNames):
        fig_row, fig_col = int(np.floor(idx/3)),idx%3
        currentAx = axs[fig_row, fig_col]
        samples = samplesDF[parameterName]
        print(f"mean {parameterName}: {np.mean(samples)}")
        currentAx.hist(samples,nBins)
        currentAx.axvline(generateDataParameters[parameterName], color='red')
        currentAx.set_title(parameterName)
    plt.show()


setup_json_path = 'setup.json'
with open(setup_json_path, 'r') as setup_file:
    setup = json.load(setup_file)
sampleEvaluation("samples/samples_1749310374.1408446.csv", setup["inferred_parameters"], setup["parameters"])
