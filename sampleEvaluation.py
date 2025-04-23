from pandas import read_csv
from matplotlib import pyplot as plt
import numpy as np

samplesCSVPath = "samples_reduced_4000_smallnoise_moredate.csv"
samplesDF = read_csv(samplesCSVPath)
print(samplesDF)

samplingResultArray = samplesDF.to_numpy()
samplesAlpha = samplesDF["p4"].to_numpy()
samplesMu = samplesDF["p8"].to_numpy()

print(f'mean alpha: {np.mean(samplesAlpha)}, mean mu: {samplesMu}')
plt.hist(samplesAlpha,50)
plt.axvline(10**(-6), color='r')
plt.show()
plt.hist(samplesMu, 50)
plt.axvline(0.1,color='r')
plt.show()