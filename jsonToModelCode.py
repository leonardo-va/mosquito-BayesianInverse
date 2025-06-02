import textwrap
import json
import quantityOfInterest
from runModel import generateData


def generate_stan_model_function(setup:dict):
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

def generate_stan_state_to_observable(setup:dict):
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

def generate_stan_initial_state(setup:dict):
    initial_values = setup["initial_state"].values()
    stan_initial_state = f"vector[{len(initial_values)}] y0Mean;"
    for idx, val in initial_values:
        stan_initial_state += f"y0Mean[{idx+1}] = {val};"
    return stan_initial_state

def generate_stan_parameters(setup:dict):
    stan_parameters = ""
    for paramName, paramVal in setup["parameters"].items():
        stan_parameters += f"real {paramName} = {paramVal};"
    return stan_parameters

def generate_stan_solver_call(setup:dict):
    relative_tolerance = 10**(-2)
    absolute_tolerance = 10**(-2)
    max_num_steps = 5000
    size_state = len(setup["initial_state"])
    solver_call = f"array[N] vector[{size_state}] mu = ode_rk45_tol(sho, y0Mean, t0, ts, {relative_tolerance}, {absolute_tolerance}, {max_num_steps}"
    for param in setup["parameters"].keys():
        solver_call += f", {param}"
    solver_call += ");"
    return solver_call

def generate_stan_function_block(setup:dict):
    sho = generate_stan_model_function(setup)
    qoi = generate_stan_state_to_observable(setup)
    functionsBlock = f"""functions\u007b
    {sho}{qoi}\u007d"""
    return functionsBlock

def generate_stan_data_block(setup:dict):
    data_point_size = len(setup["state_to_observable"])
    data_block = """data {
    int<lower=1> N;
    array[N] vector<lower=0>[""" 
    data_block += str(data_point_size)
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
    pass
 