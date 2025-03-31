# mosquito-BayesianInverse

parameters = {
    # these have somewhat stable mosquito/host ratio, around 13. use this maybe as prior means
    "delta_E" : 0.6,
    "mu_E" : 0.875,
    "beta" : 3, # unknown ( egg laying rate )
    "alpha" : 10**(-6), # infer this, (factor for square term)
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

initialConditions ={
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
