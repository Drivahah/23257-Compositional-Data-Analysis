''' Python script to solve the SIR equations with births and deaths '''

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as si

R0 = 6.
MU = 0.8/365
GAMMA = 1./3


def SIR(timestep, pop):
    ''' Basic SIR equations with demographics '''
    S, I, R = pop
    beta = R0*(MU+GAMMA)

    dS = MU - beta*S*I - MU*S
    dI = beta*S*I - GAMMA*I - MU*I
    dR = GAMMA*I - MU*R

    return [dS, dI, dR]


solution = si.solve_ivp(SIR, (0, 500), [1.-1e-2, 1e-2, 0],  method='LSODA',
                        t_eval=np.arange(0., 500., 0.05))

plt.clf()
plt.plot(solution.t, solution.y[0, :], 'orange', label='S(t)', lw=1)
plt.plot(solution.t, solution.y[1, :], 'maroon', label='I(t)', lw=1)
plt.plot(solution.t, solution.y[2, :], 'seagreen', label='R(t)', lw=1)
plt.xlabel('Time')
plt.ylabel('Proportion of population')
plt.ylim((0, 1))
plt.show()
