import stan
import time as t
import json
import os   

def sample(stan_code:str, data:dict, num_samples:int):
    posterior = stan.build(program_code=stan_code, data=data)
    num_chains = 4
    num_samples = int(num_samples/num_chains)
    fit = posterior.sample(num_chains=num_chains, num_samples=num_samples, num_warmup = 1000)
    df = fit.to_frame()
    print(df.describe().T)
    return df

def save_samples(samples_dataframe, folder_path, setup:dict = None):
    print("saving samples to: ", folder_path)
    timestamp = t.time()
    folder_path = os.path.join(folder_path, f"{timestamp}")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    samples_filename = f"samples_{timestamp}"
    samples_csv_path = os.path.join(folder_path, f"{samples_filename}.csv")
    samples_dataframe.to_csv(samples_csv_path)
    if(setup is not None):
        with open(os.path.join(folder_path, f"{samples_filename}_setup.json"), "w") as f:
            json.dump(setup, f, indent=4)
    return samples_csv_path