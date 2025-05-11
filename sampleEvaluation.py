from pandas import read_csv
from matplotlib import pyplot as plt
import numpy as np

def sampleEvaluation(samplesCSVPath, inferedParameterNames , generateDataParameters):
    samplesDF = read_csv(samplesCSVPath)
    # samplingResultArray = samplesDF.to_numpy()
    samplesAlpha = samplesDF["p4"].to_numpy()
    samplesMu = samplesDF["p8"].to_numpy()
    for parameterName in inferedParameterNames:
        samples = samplesDF[parameterName]

    print(f'mean alpha: {np.mean(samplesAlpha)}, mean mu: {samplesMu}')
    plt.hist(samplesAlpha,50)
    plt.axvline(10**(-6), color='r')
    plt.show()
    plt.hist(samplesMu, 50)
    plt.axvline(0.1,color='r')
    plt.show()

sampleEvaluation("samples_1746035771.4881766.csv")