**Solvers**

Check if the ode solvers work correctly: let it solve exponential function.
Had doubts that RK4 works correctly because there were NANs, so i implemented Euler as well because its simple and you cant go wrong there. But still NANs.
Description solver class:
main function .solve takes arguments 
* odeRHS: function that evaluates the right hand side f(t,u) of the ode u'=f(t,u)
* T: Total time to forward solve
* initial condition: a tuple (t_0, u_0)
* stepsize
* method: Currently "RK4" and "Euler". This chooses and increment function

solve iteratively calculates: nextState = CurrentState + stepsize*increment
The increment function depends on the chosen method.


**derivative evaluation** 

right now u and the result are both transposed and copied every evaluation of rhs. see if that can be avoided


**Problem with NANs:**

eggs and juveniles are set to 0 for in the initial conditions for now, since the squared term overflows and breaks everything. This has impact on mosquito population and (somehow) host population
need checking why it impacts host population. Need to implement something to adapt parameters such that mosquito population doesnt blow up.
Again, squared term is really bad and doest make sense. have to replace it by something else, and/or try with smaller 
stepsize. Juveniles go into the negative and then this gets out of control. Make alpha small works ok


**Host Population**

How to keep host population constant: Adding the derivatives of the host SEIR yields N' = LAMBDA - sum(mu_H * compartment) = LAMBDA - mu_H * N
Thus LAMBDA = mu_H * N. This has to be considered in the parameter settings. TODO: Automate this in a parameter preprocessing step, together with other dependencies in parameters. 
Or write a function that takes things like total number of hosts and gives back a parameter list.

**Quantities of Interest**

To observe things like host population: This is not directly possible from the solution, since solution only has SEIR compartments. 
So there is a quantity of interest file now that has a function "linear combination". This takes an interpolant of a solution and gives back an interpolant of a linear
combination of the compartments.
To get for example total hosts, you give the coefficients [0,0,0,0,0,1,1,1,1] which corresponds to S_h+E_h+I_h+R_h = N

**Test the model**

results make sense so far. i put 0 for eggs and juvenlies. The mosq. population dies out quickly after laying a couple of eggs. 
Infected hosts go to 0 due to mosquitos dying.

parameters = {
    "delta_E" : 0.6,
    "mu_E" : 0.875,
    "beta" : np.random.rand(),
    "alpha" : np.random.rand(),
    "delta_J" : 0.09,
    "mu_J" : np.random.rand(),
    "omega" : 0.5,
    "mu_M" : np.random.rand(),
    "a" : 0.2,
    "b_M" : 0.9,
    "alpha_M" : 1/30,
    "Lambda" : 12, # test
    "b_H" : 0.8,
    "mu_H" : 0.001, # test
    "alpha_H" : 0.4,
    "gamma_H" : 1/5.5
}

initialConditions ={
    "eggs":0,
    "juveniles":0,
    "susceptible_m":50000,
    "exposed_m":10000,
    "infected_m":10000,
    "susceptible_h":10000,
    "exposed_h":1000,
    "infected_h":1000,
    "recovered_h":0
}

plotted the mosq/host ratio. mosq die out quickly because of -J^2 term in juveniles. Found some parameter with trial and error that has the mosq. population and ratio somewhat stable long term: 
can use them as prior means.
Also the egg laying rate has to consider that there is more than 1 egg per lay. I used 0.1-0.2 as rate and that was too small(0.1 would mean there is 1 lay with 1 egg in lifecycle of 10 days. So multiply by 30). 

parameters = {
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
