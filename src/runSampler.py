import stan
import time as t
import json
import os  
import arviz as az 

def sample(stan_code:str, data:dict, num_samples:int):
    posterior = stan.build(program_code=stan_code, data=data)
    num_chains = 4
    num_samples = int(num_samples/num_chains)
    num_warmup = 1000
    start = t.perf_counter()
    fit = posterior.sample(num_chains=num_chains, num_samples=num_samples, num_warmup = num_warmup)
    end = t.perf_counter()
    sampling_time_s = end-start
    az_data = az.from_pystan(posterior=fit)
    ess_bulk = az.ess(az_data, method="bulk")
    ess_tail = az.ess(az_data, method="tail")
    # print(f"time to sample from {num_chains} chains with {num_samples} each: {sampling_time_s:.4g} s")
    # print(f"effective sample size (bulk): {ess_bulk} out of {num_samples} samples")
    # print(f"effective sample size (tails): {ess_tail} out of {num_samples} samples")
    # print(f"average time per sample: {sampling_time_s/(num_chains*num_samples):.4g}s")
    # print(f"average time per independent sample: {sampling_time_s/(num_chains*ess_bulk):.4g}")
    summary = {"num_chains" : num_chains,
               "samples_per_chain": num_samples,
               "ess_bulk": ess_bulk,
               "ess_tail": ess_tail,
               "total_time_s": sampling_time_s,
               "num_warmup": num_warmup}

    
    df = fit.to_frame()
    print(df.describe().T)
    return df, summary

# def show_summary(summary:dict):
#     print("----------------------------------\nsummary\n----------------------------------\n")
#     num_chains, num_samples, ess_bulk, ess_tail, sampling_time_s, num_warmup = summary.values()
#     print(f"time to sample from {num_chains} chains with {num_samples} samples each: {sampling_time_s:.4g} s")
#     print(f"average time per sample: {sampling_time_s/(num_chains*num_samples):.4g}s")
#     print(f"number of warmup samples per chain: {num_warmup}")
#     for param_name,  ess_bulk_val in ess_bulk.data_vars.items():
#         print(f"effective sample size (bulk) for {param_name}: {ess_bulk_val.values:.4g} out of {num_chains * num_samples} samples")
#         print(f"effective sample size (tails) for {param_name}: {ess_tail.data_vars[param_name].values:.4g} out of {num_chains * num_samples} samples")
#         print(f"average time per independent sample({param_name}): {sampling_time_s/(num_chains*ess_bulk_val.values):.4g} s")
#     print("----------------------------------\n----------------------------------\n")


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