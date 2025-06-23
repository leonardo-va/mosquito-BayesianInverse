import stan
import quantityOfInterest
from runModel import generateData
import time as t
import json
import jsonToModelCode
import os

setup_json_path = 'setup.json'

with open(setup_json_path, 'r') as setup_file:
    setup = json.load(setup_file)

ode_Code = jsonToModelCode.generate_stan_ode_code(setup)

# set your data here
ode_Data = generateData(setup["number_of_measurements"], setup["time_interval"], setup["parameters"], setup["initial_state"], 
                        [quantityOfInterest.numberOfMosquitos, 
                         quantityOfInterest.numberOfEggs])

# build stan model and sample
posterior = stan.build(ode_Code, data=ode_Data)
fit = posterior.sample(num_chains=4, num_samples=setup["number_of_samples"])

df = fit.to_frame()


samples_path = os.path.join('.', 'samples') 
if not os.path.exists(samples_path):
    os.makedirs(samples_path)
samples_filename = f"samples/samples_{t.time()}"
df.to_csv(f"{samples_filename}.csv")
with open(f"{samples_filename}_setup.json", "w") as f:
    json.dump(setup, f, indent=4)
print(df.describe().T)

 