import numpy as np
import re

def _generate_stan_model_function(setup:dict):
    declaration = """vector sho(real t, vector u"""
    for paramName in setup["parameters"].keys():
        declaration += ", real "
        declaration += paramName
    declaration += ")"

    lenght_state = len(setup["initial_state"])
    body = "{ vector[" + str(lenght_state) + "] res;"
    for idx, equation in enumerate(setup["ode_rhs"]):
        body += f"res[{idx+1}] = {equation};"
    body += "return res; }"
    stan_function = declaration + body
    return stan_function

def _generate_stan_state_to_observable(setup:dict):
    declaration = """vector qoi(vector u)"""
    body = "{" + f"vector[{len(setup['state_to_observable'])}] res;"
    for idx_equation, linearCombination in enumerate(setup['state_to_observable']):
        eq = f"res[{idx_equation+1}] = "
        for idx_coeff, coeff in enumerate(linearCombination):
            if(coeff != 0):
                eq += f"+{coeff}*u[{idx_coeff}]"
        first_plus = eq.find("+")
        if first_plus != -1:
            eq = eq[:first_plus] + eq[first_plus+1:]
        eq += ";"
        body+=eq
    body += "return res;}"
    state_to_obs_fct = declaration + body
    return state_to_obs_fct

def _generate_stan_initial_state(setup:dict):
    initial_values = setup["initial_state"].values()
    stan_initial_state = f"vector[{len(initial_values)}] initialState;"
    for idx, val in enumerate(initial_values):
        stan_initial_state += f"initialState[{idx+1}] = {val};"
    return stan_initial_state

def _generate_stan_parameters(setup:dict):
    stan_parameters = ""
    for paramName, paramVal in setup["parameters"].items():
        stan_parameters += f"real {paramName} = {paramVal};"
    return stan_parameters

def _generate_stan_solver_call(setup:dict):
    relative_tolerance = 10**(-2)
    absolute_tolerance = 10**(-2)
    max_num_steps = 5000
    size_state = len(setup["initial_state"])
    solver_call = f"array[N] vector[{size_state}] mu = ode_rk45_tol(sho, initialState, t0, ts, {relative_tolerance}, {absolute_tolerance}, {max_num_steps}"
    for param in setup["parameters"].keys():
        solver_call += f", {param}"
    solver_call += ");"
    return solver_call

def generate_stan_function_block(setup:dict):
    sho = _generate_stan_model_function(setup)
    qoi = _generate_stan_state_to_observable(setup)
    functionsBlock = f"""functions\u007b
    {sho}{qoi}\u007d"""
    return functionsBlock

def generate_stan_data_block(setup:dict):
    number_observables = len(setup["state_to_observable"])
    data_block = """data {
    int<lower=1> N;
    array[N] vector<lower=0>[""" 
    data_block += str(number_observables)
    data_block += """] y;
    real t0;
    array[N] real ts;
    }"""
    return data_block

def generate_stan_parameters_block(setup:dict):
    parameters_block = """parameters {"""
    for parameter in setup["inferred_parameters"].keys():
        parameters_block += f"real<lower=0> {parameter};"
    parameters_block += "}"
    return parameters_block

def generate_stan_model_block(setup:dict):
    fixed_parameters = [parameter for parameter in setup['parameters'] if parameter not in setup['inferred_parameters']]
    model_block = """model {"""
    
    model_block_initial_state = _generate_stan_initial_state(setup=setup)
    
    model_block_fixed_parameters = ""
    for fixed_parameter in fixed_parameters:
        value = setup['parameters'][fixed_parameter]
        model_block_fixed_parameters += f"real {fixed_parameter} = {value};"

    model_block_solver_call = _generate_stan_solver_call(setup=setup)

    model_block_priors = ""
    for inferred_parameter, distribution in setup['inferred_parameters'].items():
        model_block_priors += f"{inferred_parameter} ~ {distribution[0]}("
        for distribution_parameter in distribution[1:]:
            model_block_priors += f"{distribution_parameter},"  
        if model_block_priors.endswith(","):
            model_block_priors = model_block_priors[:-1]
        model_block_priors += f") T[0,1];"
    
    model_block_noise = ""
    observables = setup["state_to_observable"]
    number_observables = len(observables)
    observable_standard_deviation = setup["observable_standard_deviation"]
    model_block_noise += f"array[N] vector[{number_observables}] q;"
    model_block_noise += "for(t in 1:N){q[t] = qoi(mu[t]);}"
    model_block_noise += f"for(t in 1:N)" + "{" + f"y[t] ~ normal(q[t], {observable_standard_deviation**2});" + "}"
    
    model_block += model_block_initial_state
    model_block += model_block_fixed_parameters
    model_block += model_block_solver_call
    model_block += model_block_priors
    model_block += model_block_noise
    model_block += "}"
    return model_block

def generate_stan_ode_code(setup:dict):
    '''
    Generates and returns the stan program code string from the setup.json.
    Stan can then compile this.
    '''
    functionsBlock = generate_stan_function_block(setup)
    dataBlock = generate_stan_data_block(setup)
    parametersBlock = generate_stan_parameters_block(setup)
    modelBlock = generate_stan_model_block(setup)
    ode_code = functionsBlock + dataBlock + parametersBlock + modelBlock
    
    return ode_code

def generate_py_model_function(setup:dict):
    '''
    Generates and returns a python function from the \"parameters\" and \"ode_rhs\" fields in the setup.json, that evaluates the ode right hand side.
    The generated function arguments are: func(t, u : np.array, parameters : np.array)->np.array
    '''
    # the model function takes a parameters vector, but the equations in the json use parameter names
    # Replace the parameter names with the expression parameters[index]
    equations_with_param_names = setup["ode_rhs"]
    equations_with_param_index = []
    for equation in equations_with_param_names:
        equation_with_param_index = equation
        for idx, pname in enumerate(setup["parameters"].keys()):
            pattern = rf'(?<![a-zA-Z0-9_]){re.escape(pname)}(?![a-zA-Z0-9_])'
            replacement = f'parameters[{idx}]'
            equation_with_param_index = re.sub(pattern, replacement, equation_with_param_index)
  
        equation_with_param_index = equation_with_param_index.replace("^", "**")
        equations_with_param_index.append(equation_with_param_index)
    
    # indices in json equation start from 1 (because stan does so). For the python model, they 
    # need to be reduced by 1
    for idx,equation in enumerate(equations_with_param_index):
        pattern = r'u\[(\d+)\]'
        equation = re.sub(pattern, lambda match: f'u[{int(match.group(1))-1}]', equation)
        equations_with_param_index[idx] = equation
    
    # build a string that defines the model function
    modelFuncString = """def model_function(t, u, parameters):\n\tu = u.T\n\tres = np.zeros((1,len(u))).squeeze()"""
    for idx, equation in enumerate(equations_with_param_index):
        modelFuncString += "\n\t"
        modelFuncString += f"res[{idx}] = {equation}"
    modelFuncString += "\n\treturn res"
    
    # execute the model function definition, store it in a local namespace so it can be accessed
    modelFuncNamespace = {}
    exec(modelFuncString, {'np':np}, modelFuncNamespace)
    model_function = modelFuncNamespace['model_function']
    
    return model_function
