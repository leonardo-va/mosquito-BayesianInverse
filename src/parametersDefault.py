import numpy as np
import json

def generateDefaultSetup(setup_file_path):
    default_setup = {
    "parameters":{
        "delta_E" : 0.6,
        "mu_E" : 0.875,
        "beta" : 3, 
        "alpha" : 0.000002, 
        "delta_J" : 0.09,
        "mu_J" : 0.1, 
        "omega" : 0.5,
        "mu_M" : 0.1, 
        "a" : 0.2,
        "b_M" : 0.9,
        "alpha_M" : 0.0333,
        "Lambda" : 12, 
        "b_H" : 0.8,
        "mu_H" : 0.001,
        "alpha_H" : 0.4,
        "gamma_H" : 0.1818
    },

    "ode_rhs":[
        "beta*(u[3]+u[4]+u[5]) - (delta_E+mu_E)*u[1]",
        "delta_E*u[1] - alpha*(u[2]^2) - mu_J*u[2] - delta_J*u[2]",
        "omega*delta_J*u[2] - a*b_M*u[3]*u[8]/(u[6]+u[7]+u[8]+u[9]) - mu_M*u[3]",
        "a*b_M*u[3]*u[8]/(u[6]+u[7]+u[8]+u[9]) - (alpha_M+mu_M)*u[4]",
        "alpha_M*u[4] - mu_M*u[5]",
        "Lambda - a*b_H*u[5]*u[6]/(u[6]+u[7]+u[8]+u[9]) - mu_H*u[6]",
        "a*b_H*u[5]*u[6]/(u[6]+u[7]+u[8]+u[9]) - alpha_H*u[7] - mu_H*u[7]",
        "alpha_H*u[7]-gamma_H*u[8]-mu_H*u[8]",
        "gamma_H*u[8] - mu_H*u[9]"
    ],

    "initial_state":{
        "eggs":10000,
        "juveniles":10000,
        "susceptible_m":50000,
        "exposed_m":10000,
        "infected_m":10000,
        "susceptible_h":10000,
        "exposed_h":1000,
        "infected_h":1000,
        "recovered_h":0
    },

    "time_interval": [0,10],
    
    "number_of_measurements": 100,
    
    "inferred_parameters":{
        "alpha": ["lognormal",-13.1,0.5],
        "mu_M": ["normal",0.1,0.1]
    },

    "state_to_observable":[
        [0,0,1,1,1,0,0,0,0],
        [0,1,0,0,0,0,0,0,0]
    ],

    "observable_standard_deviation": 5000,

    "number_of_samples": 1000
    }
    with open(setup_file_path, "w") as default_file:
        json.dump(default_setup, default_file, indent=4)
    return default_setup


defaultParameters = {
    # these have somewhat stable mosquito/host ratio, around 13. use this as prior means
    "delta_E" : 0.6,
    "mu_E" : 0.875,
    "beta" : 3, # unknown ( egg laying rate )
    "alpha" : 2*10**(-6), # infer this, (factor for square term)
    "delta_J" : 0.09,
    "mu_J" : 0.1, # unknown, juvenile mortality
    "omega" : 0.5,
    "mu_M" : 0.1, # infer this, (mosquito mortality rate)
    "a" : 0.2,
    "b_M" : 0.9,
    "alpha_M" : 1/30,
    "Lambda" : 12, # uknown
    "b_H" : 0.8,
    "mu_H" : 0.001, # uknown
    "alpha_H" : 0.4,
    "gamma_H" : 1/5.5
}

defaultInitialConditions ={
    "eggs":10000,
    "juveniles":10000,
    "susceptible_m":50000,
    "exposed_m":10000,
    "infected_m":10000,
    "susceptible_h":10000,
    "exposed_h":1000,
    "infected_h":1000,
    "recovered_h":0
}


