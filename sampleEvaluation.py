from pandas import read_csv

samplesCSVPath = "/home/leo/src/mosquito-BayesianInverse/samples_reduced_4000_large_noise.csv"
samplesDF = read_csv(samplesCSVPath)
print(samplesDF.keys())

samplingResultArray = samplesDF.to_numpy()
samplesAlpha = samplesDF["p4"].to_numpy()
samplesMu = samplesDF["p8"].to_numpy()
print(samplesMu)