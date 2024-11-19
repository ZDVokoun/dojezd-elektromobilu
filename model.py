import numpy as np
from numpy.linalg import inv


class BayesReg:
    def __init__(self, row_size, delay = 0, intercept = True):
        self.n_of_variables = row_size
        self.delay = delay
        self.reg_vec_size = (row_size + 1) * (delay + 1) - 1
        self.intercept = intercept
        if intercept:
            self.reg_vec_size += 1
        self.Eth = 0.01 * np.ones(self.reg_vec_size)
        self.r = 1
        self.V = 1e-8 * np.eye(self.reg_vec_size + 1, self.reg_vec_size + 1)
        self.delayed_val = np.zeros((self.n_of_variables + 1) * self.delay)
        self.k = 1

    def get_reg_vec(self, x, y=None):
        """Get regression vector."""
        Ps = np.concatenate((x, self.delayed_val))
        if y is not None:
            Ps = np.concatenate(([y], Ps))
        if self.intercept:
            Ps = np.concatenate((Ps, [1]))
        return Ps

    def get_state_vec(self):
        return np.concatenate((self.delayed_val[2:self.n_of_variables + 1], self.delayed_val, [1]))

    def get_theta(self):
        return self.Eth

    def get_r(self):
        return self.r

    def deleteDelayed(self) -> None:
        self.delayed_val = np.zeros((self.n_of_variables + 1) * self.delay)

    def updateDelayed(self,x,y) -> None:
        Ps = self.get_reg_vec(x,y)
        self.delayed_val = Ps[0:(self.n_of_variables + 1) * self.delay]

    def updateK(self, /, weight = 1) -> None:
        self.k += weight

    def updateV(self, x, y, /, weight = 1) -> None:
        Ps = self.get_reg_vec(x,y)
        self.V += weight * np.outer(Ps, Ps)
        self.updateDelayed(x,y)

    def updateParams(self) -> None:
        Vp = self.V[1:, 1:]
        Vyp = self.V[1:, 0]
        self.Eth = np.linalg.inv(Vp + 1e-6 * np.eye(Vp.shape[0])) @ Vyp
        self.r = (self.V[0,0] - Vyp.T @ np.linalg.inv(Vp + 1e-6 * np.eye(Vp.shape[0])) @ Vyp) / self.k

    def predict(self, x, updateDelayed=False):
        ps = self.get_reg_vec(x)
        y = np.squeeze(ps.dot(self.Eth))
        if updateDelayed:
            self.updateDelayed(x,y)
        return y

    def predict_update(self, x, y):
        res = self.predict(x)
        self.updateV(x,y)
        self.updateParams()
        return res


    # def get_M_N2(self):
    #     """
    #     Returns matrices in space-form model x_t = Mx_{t-1} + Nu_t + \\omega
    #
    #     For all models of second or higher order, we model all
    #     variables except of predicted and controled variables as a
    #     random walk.
    #     """
    #     M_size = (self.n_of_variables + 1) * self.delay + 1
    #
    #     M = np.zeros((M_size, M_size))
    #     first_row = self.Eth[self.n_of_variables:]
    #     if not self.intercept:
    #         first_row = np.concatenate((first_row, [0.0]))
    #
    #     M[0] = first_row
    #     M[2:self.n_of_variables + 1, 0:self.n_of_variables + 1] = np.eye((self.n_of_variables + 1))[2:,:]
    #     M[self.n_of_variables + 1: -1, 0: -2 - self.n_of_variables] = np.eye((self.n_of_variables + 1) * (self.delay - 1))
    #     M[-1,-1] = 1
    #
    #     N = np.zeros((M_size, 1))
    #     N[0] = self.Eth[0]
    #     N[1] = 1
    #     return M, N
    
    def get_M_N(self):
        """
        Returns matrices in space-form model x_t = Mx_{t-1} + Nu_t + \\omega

        For all models of second or higher order, we model all
        variables except of predicted and controled variables as a
        random walk.
        """
        M_size = (self.n_of_variables + 1) * self.delay + self.n_of_variables

        M = np.zeros((M_size, M_size))
        first_row = self.Eth[1:]
        if not self.intercept:
            first_row = np.concatenate((first_row, [0.0]))

        n = self.n_of_variables

        # for v_{t+1}
        M[0:n - 1, 0:n - 1] = np.eye(n - 1)
        # for y_t
        M[n - 1] = first_row
        # for u_t ... row of zeros
        # for the rest except the last row
        M[n+1:-1, 0:n - 1 + (n + 1) * (self.delay - 1)] = np.eye(n - 1 + (n + 1) * (self.delay - 1))
        # last row
        M[-1,-1] = 1

        N = np.zeros((M_size, 1))
        N[self.n_of_variables - 1] = self.Eth[0]
        N[self.n_of_variables] = 1
        return M, N
    
    def optimal_control(self,n, om, la, setpoint, ka = None, upper = None, lower = None):
        """
        Optimal control algorithm. The error function is
        $J_t = (y_t - s_t)^2 \\omega + (u_t - u_{t - 1})^2 \\lambda + (u - u_t) (u_t - d) \\kappa$

        For all models of second or higher order, we model all
        variables except of predicted and controled variables as a
        random walk.
        """
        M, N = self.get_M_N()

        if type(setpoint) is float or type(setpoint) is int:
            setpoint = np.ones(n) * setpoint

        Omega = np.zeros((M.shape[0], M.shape[0]))
        Omega[self.n_of_variables - 1, self.n_of_variables - 1] = om
        if la != 0:
            Omega[self.n_of_variables,self.n_of_variables] = la
            Omega[2 * self.n_of_variables + 1, 2 * self.n_of_variables + 1] = la
            Omega[self.n_of_variables, 2 * self.n_of_variables + 1] = -la
            Omega[2 * self.n_of_variables + 1, self.n_of_variables] = -la
        # if ka is not None:
        #     Omega[self.n_of_variables,self.n_of_variables] += ka
        #     Omega[-1,self.n_of_variables] += -ka * upper
        #     Omega[self.n_of_variables,-1] += -ka * lower
        #     Omega[-1,-1] += ka * upper * lower
        # Omega[1:4, 1:4] = np.array([[la, 0, -la], [0, 0, 0], [-la, 0, la]])

        R = np.zeros((M.shape[0], M.shape[0]))
        u = np.zeros(n)
        y = np.zeros(n)
        S = [None] * n

        for i in range(n - 1, -1, -1):
            if setpoint[i] != 0:
                Omega[self.n_of_variables - 1,-1] = -om * setpoint[i]
                Omega[-1,self.n_of_variables - 1] = -om * setpoint[i]
                Omega[-1,-1] = om * setpoint[i]**2
                # if ka is not None:
                #     Omega[-1,-1] += -ka

            U = Omega + R
            A = N.T @ U @ N
            B = N.T @ U @ M
            C = M.T @ U @ M
            S[i] = inv(A + np.eye(A.shape[0]) * 1e-8) @ B
            R = C - S[i].T @ A @ S[i]

        # x = self.get_state_vec()
        # for i in range(0, n):
        #     u[i] = -S[i] @ x
        #     x = M @ x + N[:,0] * u[i]
        #     y[i] = self.predict(x[1:self.row_size+1], updateDelayed=True)
        #     x[0] = y[i]
        x = self.get_state_vec()
        for i in range(0, n):
            # print(x)
            u[i] = -S[i] @ x
            u[i] = min(upper, u[i])
            u[i] = max(lower, u[i])
            x = M @ x + N[:,0] * u[i]
            y[i] = x[self.n_of_variables - 1]

        return u,y

    def optimal_control2(self,n, om = 1.0, la = 0.0, setpoint = 0):
        M, N = self.get_M_N()

        if type(setpoint) is float or type(setpoint) is int:
            setpoint = np.ones(n) * setpoint

        A = np.eye(M.shape[0]) * 1e-8
        B = np.eye(1) * 1e-8
        C = np.zeros((M.shape[0],1))
        D = np.zeros((M.shape[0],1))
        E = np.zeros((1,1))
        # F = 0
        Om = np.zeros((M.shape[0], M.shape[0]))
        Om[0,0] = om
        La = np.zeros((1,1))
        La[0,0] = la
        u = np.zeros(n + 1)
        y = np.zeros(n)
        Sl = [None] * n
        Wl = [None] * n
        Ul = [None] * n

        for i in range(n - 1, -1, -1):
            St = np.zeros((M.shape[0],1))
            St[0,0] = setpoint[i]
            R = M.T @ (A + Om) @ M
            S = M.T @ ((A + Om) @ N + C)
            W = inv(N.T @ A @ N + 2 * N.T @ C + B + La)
            U = N.T @ D + E - N.T @ Om @ St
            V = M.T @ (D - Om @ St)
            A = R - S @ W @ S.T
            B = La - La.T @ W @ La
            C = S @ W @ La
            D = V - S @ W @ U
            E = La.T @ W @ U
            # F = G - U.T @ W @ U
            Sl[i] = S
            Wl[i] = W
            Ul[i] = U



        for i in range(0, n):
            x = self.get_state_vec()
            u[i + 1] = -Wl[i] @ (Sl[i].T @ x - la * u[i] + Ul[i])
            new_x = np.concatenate(([u[i + 1]], x[2:self.n_of_variables+1]))
            y[i] = self.predict(new_x, updateDelayed=True)

        return u[1:],y


# class BayesRegMix:
#     def __init__(self, nc, nf, delay = 0, intercept = True) -> None:
#         self.components = []
#         self.nc = nc
#         for _ in range(nc):
#             self.components.append(BayesReg(nf, delay, intercept))
#
#     def get_proximity(self, y):
#         proximities = np.zeros(nc)
#         return 1/np.sqrt(2 * np.pi * self.r) * np.exp(-(y - self))
#         pass

