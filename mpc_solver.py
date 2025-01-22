'''
Module for constructing MPC feedback function for 
Nonlinear systems of the form:
\dot{x} = f(x,u,w)
'''
import numpy as np 
from casadi import MX, Function, nlpsol, vertcat, reshape, jacobian, if_else, power
import time, copy
import matplotlib.pyplot as plt 

class Container:
    pass 

class MPC:

    def __init__(self, nx, nu, nw, npar, 
                    ode, stage_cost, 
                    terminal_cost, 
                    constraints, umin, umax, 
                    gmin, gmax, dt, N, max_iter):

        self.nx = nx
        self.nu = nu 
        self.nw = nw 
        self.N = N
        self.dt = dt
        self.npar = npar
        self.ode = ode 
        self.stage_cost = stage_cost
        self.terminal_cost = terminal_cost
        self.constraints = constraints
        self.max_iter = max_iter

        # create discrete time version of the ode 
        # this is the one-step-ahead-predictor

        x = MX.sym('x', nx)
        u = MX.sym('u', nu)
        w = MX.sym('w', nw)
        p = MX.sym('p', npar)
        tau = MX.sym('tau')
        k1 = ode(x, u, w, p)
        k2 = ode(x + 0.5 * k1 * tau, u, w, p)
        k3 = ode(x + 0.5 * k2 * tau, u, w, p)
        k4 = ode(x + k3 * tau, u, w, p)
        xplus = x +tau/6*(k1 + 2 * (k2+k3) + k4)
        self.F = Function('F', [x, u, w, p, tau], [xplus])

        # create the multi-step ahead predictor

        def sim(x0, U, W, p, Nsim):

            X = x0
            x = x0
            for i in range(Nsim):
                u = U[i*nu : (i+1) * nu]
                w = W[i*nw : (i+1) * nw]
                x = self.F(x, u, w, p, dt)
                X = vertcat(X, x)

            return X

        self.sim = sim

        # Create the optimization problem and the bounds on the constraints

        x0 = MX.sym('x0', nx)
        U = MX.sym('U', N * nu)
        W = MX.sym('W', N * nw)
        par = MX.sym('par', npar)
        g = []
        J = 0
        x = x0
        for i in range(N):
            u = U[i * nu : (i+1) * nu]
            w = W[i * nw : (i+1) * nw]
            J += stage_cost(x, u, w, par) * dt
            gi = constraints(x, u, w, par)
            g = vertcat(g, gi)
            x = self.F(x, u, w, par, dt)
        J += terminal_cost(x, w, par)

        self.problem = {'f':J, 'x':U, 'g':vertcat(g), 'p':vertcat(x0, W, par)}
        
        self.Umin = np.array(list(umin) * N).flatten()
        self.Umax = np.array(list(umax) * N).flatten()
        self.gmin = np.array(list(gmin) * N).flatten()
        self.gmax = np.array(list(gmax) * N).flatten()

        # create the function that computes the gradient 
        # of the softened cost function 

        rho_soft = MX.sym('rho_soft')
        Jsoft = J
        for i in range(len(self.gmin)):
            term = g[i]-self.gmax[i]
            Jsoft += if_else(term<0, 0, rho_soft * power(term, 2))
            term = self.gmin[i]-g[i]
            Jsoft += if_else(term<0, 0, rho_soft * power(term, 2))
        
        self.cost = Function('cost', [U, x0, W, par], [J])
        self.soft_cost = Function('soft_cost', [U, x0, W, par, rho_soft], [Jsoft])

        self.solver = nlpsol('solver', 'ipopt', 
                            self.problem, {'ipopt': {'max_iter':max_iter}})
        
        self.compute_MPC = lambda U0, x0, W, par: self.solver(x0=U0, lbx=self.Umin, 
                    ubx=self.Umax, lbg=self.gmin, ubg=self.gmax, p=vertcat(x0, W, par))

        G = jacobian(Jsoft, U)
        
        self.grad = Function('grad', [U, x0, W, par, rho_soft], [G])
        
    def simulate_cl(self, x0, U0, W_supposed, W_real, p, Nsim):

        Xcl = x0
        Ucl = []
        Jcl = [self.cost(U0, x0, W_supposed[0:self.N * self.nw], p).full().flatten()[0]]
        cpu = []
        x = x0
        for k in range(Nsim-self.N):
            W = W_supposed[k * self.nw: (k+self.N) * self.nw]
            if k == 0:
                U0_guess = U0
            else:
                U0_guess = Uopt

            t0 = time.time()
            sol = self.compute_MPC(U0_guess, x, W, p)
            cpu += [time.time()-t0]
            Uopt = sol['x'].full().flatten()
            Jcl = vertcat(Jcl, sol['f'])
            u = Uopt[0:self.nu]
            w = W_real[k * self.nw: (k+1) * self.nw]
            x = self.F(x, u, w, p, self.dt)
            Xcl = vertcat(Xcl, x)
            Ucl = vertcat(Ucl, u)

        Xcl = Xcl.full().flatten().reshape(-1, self.nx)
        Ucl = Ucl.full().flatten().reshape(-1, self.nu)
        Jcl = Jcl.full().flatten()
        tcl = np.array([i * self.dt for i in range(len(Xcl))])
        Ucl = np.array(list(Ucl)+[Ucl[-1]])
        cpu = np.array(cpu+[cpu[-1]])

        Result = Container()
        Result.tcl = tcl 
        Result.Xcl = Xcl 
        Result.Ucl = Ucl 
        Result.Jcl = Jcl 
        Result.cpu = cpu

        return Result

    def fast_gradient(self, U, x, W, p, rho_soft, gam, c, Niter):

        # The fast gradient iteration
        # U is the current guess that to be initialized
        # to U = Uopt_previous 

        def project(U):

            U_projected = copy.copy(np.array(U.flatten()))
            for i in range(self.N * self.nu):
                    if U_projected[i] > self.Umax[i]:
                        U_projected[i] = self.Umax[i]
                    elif U_projected[i] < self.Umin[i]:
                        U_projected[i] = self.Umin[i]
            return U_projected

        Uopt_previous = copy.copy(U)
        z_previous = copy.copy(U)

        for _ in range(Niter):
            G = self.grad(Uopt_previous, x, W, p, rho_soft).full().flatten()
            z = Uopt_previous - gam * G
            Uopt = project(z + c * (z-z_previous))
            z_previous = copy.copy(z)
            Uopt_previous = copy.copy(Uopt)

        return Uopt

    def simulate_cl_fast_gradient(self, x0, U0, W_supposed, 
                                    W_real, p, rho_soft, Nsim, gam, c, Niter):

        Xcl, Ucl, cpu, x, Jcl = x0, [], [], x0, []
    
        for k in range(Nsim-self.N):
        
            W = W_supposed[k * self.nw: (k+self.N) * self.nw]
            
            if k==0:
                J = self.soft_cost(U0, x, W, p, rho_soft).full().flatten()[0]
                Jcl = vertcat(Jcl, J)

            else:
                J = self.soft_cost(Uopt, x, W, p, rho_soft).full().flatten()[0]
                Jcl = vertcat(Jcl, J)
            t0 = time.time()
           
            if k==0:
                Uopt = copy.copy(U0)

            Uopt = self.fast_gradient(Uopt, x, W, p, rho_soft, gam, c, Niter)
            u = Uopt[0:self.nu]
            cpu += [time.time()-t0]
        
            w = W_real[k * self.nw: (k+1) * self.nw]
            x = self.F(x, u, w, p, self.dt)
            Xcl = vertcat(Xcl, x)
            Ucl = vertcat(Ucl, u)

        Xcl = Xcl.full().flatten().reshape(-1, self.nx)
        Ucl = Ucl.full().flatten().reshape(-1, self.nu)
        Jcl = Jcl.full().flatten()
        tcl = np.array([i * self.dt for i in range(len(Xcl))])
        Ucl = np.array(list(Ucl)+[Ucl[-1]])
        Jcl = np.array(list(Jcl)+[Jcl[-1]])
        cpu = np.array(cpu+[cpu[-1]])

        Result = Container()
        Result.tcl = tcl 
        Result.Xcl = Xcl 
        Result.Ucl = Ucl 
        Result.Jcl = Jcl 
        Result.cpu = cpu
        
        return Result

    def plot(self, R):
    
        tcl = R.tcl 
        Xcl = R.Xcl
        Ucl = R.Ucl 
        Jcl = R.Jcl 
        cpu = R.cpu 

        plt.figure(figsize=(12,6))
        #-----
        plt.plot(tcl, Xcl)
        plt.xlabel('Time')
        plt.title('Closed-loop evolution of states')
        plt.xlim([tcl.min(), tcl.max()])
        plt.grid(True)
        plt.legend([f'x{i+1}' for i in range(Xcl.shape[1])])
        plt.show()
        #-----
        plt.figure(figsize=(12,6))
        plt.step(tcl, Ucl)
        plt.xlabel('Time')
        plt.title('Closed-loop Control')
        plt.xlim([tcl.min(), tcl.max()])
        plt.grid(True)
        plt.legend([f'u{i+1}' for i in range(Ucl.shape[1])])
        plt.show()
        #-----
        plt.figure(figsize=(12,6))
        plt.step(tcl, cpu)
        plt.xlabel('Time (sec)')
        plt.title('cpu (sec) time to solve optimal control problems')
        plt.xlim([tcl.min(), tcl.max()])
        plt.grid(True)
        plt.show()


            



