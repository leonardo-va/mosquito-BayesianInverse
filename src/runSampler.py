import stan
import time as t
import json
import os   

def sample(stan_code:str, data:dict, num_samples:int):
    posterior = stan.build(program_code=stan_code, data=data)
    fit = posterior.sample(num_chains=4, num_samples=num_samples)
    df = fit.to_frame()
    print(df.describe().T)
    return df

def save_samples(samples_dataframe, folder_path, setup:dict = None):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    samples_filename = f"samples/samples_{t.time()}"
    samples_dataframe.to_csv(f"{samples_filename}.csv")
    if(setup is not None):
        with open(f"{samples_filename}_setup.json", "w") as f:
            json.dump(setup, f, indent=4)