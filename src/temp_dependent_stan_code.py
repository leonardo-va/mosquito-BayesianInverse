import jsonToModelCode
from temperature_dependent_parameters import tempDependentParameterMeta

def generate_ode_rhs(setup:dict, param_meta_list:list[tempDependentParameterMeta]):
    declaration = """vector ode_rhs(real t, vector u"""
    declaration_params = ""
    for param_meta in param_meta_list:
        if(param_meta.function_type == "constant"):
            declaration_params += f", real {param_meta.stan_entries()['stan_param_name']}"
        else:
            declaration_params += f", vector {param_meta.stan_entries()['stan_param_name']}"

    params_eval = "real T = t;\n\t"
    for param_meta in param_meta_list:
        if(param_meta.function_type != "constant"):
            params_eval += f"real {param_meta.name} = {param_meta.stan_entries()['ode_rhs_access']};\n\t"

    length_state = len(setup["initial_state"])
    body = ""
    body += "vector[" + str(length_state) + "] res;\n\t"
    for idx, equation in enumerate(setup["ode_rhs"]):
        body += f"res[{idx+1}] = {equation};\n\t"
    body += "return res; \n\t}\n\t"
    stan_function = declaration + body
    stanfc = f"""
    vector ode_rhs(real t, vector u {declaration_params}){{
        {params_eval}
        {body}
    """

    return stanfc

def assemble_function_block(setup:dict, param_meta_list:list[tempDependentParameterMeta]):
    function_block_entries = [generate_ode_rhs(setup, param_meta_list),
                              jsonToModelCode._generate_stan_state_to_observable(setup)]
    # function_block_entries = [jsonToModelCode._generate_stan_model_function(setup),
    #                         jsonToModelCode._generate_stan_state_to_observable(setup)]
    for param_t in param_meta_list:
        if(param_t.function_type == "constant"): continue
        function_block_entries.append(param_t.stan_entries()['functions'])
    functions_block = f"""functions\u007b\n\t"""
    for entry in function_block_entries:
        functions_block += f"{entry}\n\t"
    functions_block += f"\n\u007d\n"
    return functions_block

def assemble_data_block(setup:dict, param_meta_list:list[tempDependentParameterMeta]):
    number_observables = len(setup["state_to_observable"])
    fixed_params_str = ""
    for param_meta in param_meta_list:
        if(param_meta.function_type == "constant" and param_meta.name not in setup['inferred_parameters'].keys()):
            fixed_params_str += f"\treal {param_meta.name};\n"
    data_block = f"""
    data {{
        int <lower=1> N_obs;
        int <lower=1> lenState;
        array[N_obs] vector<lower=0>[lenState] y;
        real t0;
        vector[N_obs] ts;
        vector[lenState] initialState;
    {fixed_params_str}
    }}
    """
    return data_block

def assemble_parameter_block(setup:dict, param_meta_list:list[tempDependentParameterMeta]):
    param_list = f""
    for param_meta in param_meta_list:
        if param_meta.name in setup['inferred_parameters']:
            if param_meta.function_type == "constant":
                param_list += f"\treal {param_meta.name};\n"
            else:
                param_list += f"\tvector[{param_meta.function_type_n_params[param_meta.function_type]}] {param_meta.stan_entries()['stan_param_name']};\n"
    parameter_block = f"""
    parameters {{
    {param_list}
    }}
    """       
    return parameter_block

def assemble_model_block(setup:dict, param_meta_list:list[tempDependentParameterMeta]):
    fixed_parameters = [parameter for parameter in setup['parameters'] if parameter not in setup['inferred_parameters']]
    model_block = """model {\n\t"""
    


    relative_tolerance = 10**(-2)
    absolute_tolerance = 10**(-2)
    max_num_steps = 500
    size_state = len(setup["initial_state"])
    solver_call = f"\n\tarray[N] vector[{size_state}] mu = ode_rk45_tol(sho, initialState, t0, ts, {relative_tolerance}, {absolute_tolerance}, {max_num_steps}"
    for param in param_meta_list:
        solver_call += f", {param.stan_entries['stan_param_name']}"
    solver_call += ");\n\t"
    

    model_block_priors = "\n\t"
    for inferred_parameter, prior in setup['inferred_parameters'].items():
        model_block_priors += f"{inferred_parameter} ~ {prior['distribution']}("
        for distribution_parameter in prior['parameters']:
            model_block_priors += f"{distribution_parameter},"  
        if model_block_priors.endswith(","):
            model_block_priors = model_block_priors[:-1]
        model_block_priors += f");\n\t"
        if "bounds" in prior:
            # model_block_priors = model_block_priors[:-1]
            model_block_priors = model_block_priors.rsplit(";",1)[0]
            model_block_priors += f" T[{prior['bounds'][0]},{prior['bounds'][1]}];\n\t"
    
    model_block_noise = "\n\t"
    observables = setup["state_to_observable"]
    number_observables = len(observables)
    
    observable_standard_deviations = setup["observable_standard_deviation"]
    stddev_list = f"["
    for idx,sigma in enumerate(observable_standard_deviations):
        stddev_list += f"{sigma},"
    stddev_list = stddev_list[:-1]
    stddev_list +="]"
    model_block_noise += f"vector[{number_observables}] sig = {stddev_list}';\n\t"
    model_block_noise += f"array[N] vector[{number_observables}] q;\n\t"
    model_block_noise += "for(t in 1:N){\n\t\tq[t] = qoi(mu[t]);\n\t}"
    model_block_noise += f"\n\tfor(t in 1:N)" + "{\n\t\t" + f"y[t] ~ normal(q[t], sig);\n\t" + "}\n"
    
    model_block += model_block_initial_state
    model_block += model_block_fixed_parameters
    model_block += model_block_solver_call
    model_block += model_block_priors
    model_block += model_block_noise
    model_block += "}"
    return model_block




