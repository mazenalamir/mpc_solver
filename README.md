## mpc_solver
A python-Casadi-Based module for nonlinear MPC. 

This module is described in the Springer book entitled: 

**Nonlinear Control of Uncertain Systems, Conventional and learning-based alternatives with Python** 

that will be released shortly in 2025.
The module enables to create the MPC feedback, simulate the resulting closed-loop system and plot the resulting trajectories in closed-loop. The robustness against parametric uncertainties can also be simulated by using different *de-tuned* vector of parameters in the MPC solver that is different from the one used int he simualted system. 

Two different solvers can be used, namely: 

- The IPOPT solver (interior point)
- The Fast Gradient solver.

Regarding the second option, the gradient is automatically computed using the `jacobian` utilitiy provided by the `casadi` framework. 

### Description of the files 

The `jupyter notebook` entitled 
