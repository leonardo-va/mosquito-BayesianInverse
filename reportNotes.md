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


**Problem with NANs:**

eggs and juveniles are set to 0 for in the initial conditions for now, since the squared term overflows and breaks everything. This has impact on mosquito population and (somehow) host population
need checking why it impacts host population. Need to implement something to adapt parameters such that mosquito population doesnt blow up.


**Host Population**

How to keep host population constant: Adding the derivatives of the host SEIR yields N' = LAMBDA - sum(mu_H * compartment) = LAMBDA - mu_H * N
Thus LAMBDA = mu_H * N. This has to be considered in the parameter settings. TODO: Automate this in a parameter preprocessing step, together with other dependencies in parameters. 
Or write a function that takes things like total number of hosts and gives back a parameter list.

**Quantities of Interest**

To observe things like host population: This is not directly possible from the solution, since solution only has SEIR compartments. 
So there is a quantity of interest file now that has a function "linear combination". This takes an interpolant of a solution and gives back an interpolant of a linear
combination of the compartments.
To get for example total hosts, you give the coefficients [0,0,0,0,0,1,1,1,1] which corresponds to S_h+E_h+I_h+R_h = N

