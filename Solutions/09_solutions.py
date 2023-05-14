import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pycoda as coda
from scipy.stats.mstats import gmean

print('\nExercise 9.1')

# First define power and perturbation functions


def power(alpha, x):
    return [pow(x[i], alpha) for i in range(len(x))]


def perturbation(a, b):
    return [a[i]*b[i] for i in range(len(a))]


# Initial composition
x0 = np.array([150, 30, 120])
# Half-lives (in billion years)
t_half = np.array([4.468, 14.05, 1.277])
# Converted to decay rates
lam = np.log(2)/t_half

# Time vector (in Billion years)
time = np.arange(0, 50, .1)

# Solve function 9.4 from the notes. Notice minus lambda because decay, not growth.
x = [perturbation(x0, power(t, np.exp(-lam))) for t in time]
x = pd.DataFrame(x).coda.closure(100)
x.columns = ['238U', '232Th', '40K']

# Plot result
plt.clf()
plt.plot(time, x['238U'], color='orange', lw=2)
plt.plot(time, x['232Th'], color='steelblue', lw=2)
plt.plot(time, x['40K'], color='seagreen', lw=2)
plt.ylabel('Composition')
plt.xlabel('Time (1e9 Years)')
plt.show()


print('\nExercise 9.2')

# Define data from exercise
A = np.array([[0.56,  2.55, -3.11], [-1.4, -1.61,  3.01], [0.84, -0.94,  0.1]])
f = np.array([0.37, 0.03, 0.6])

balances = np.array([[1, 1, -1], [1, -1, 0]])
psi = coda.extra.norm(balances)

# Calculate contrast matrix
Astar = np.dot(psi, np.dot(A, psi.T))
print("Contrast matrix:", Astar)
# Calculate eigenvalues
eigvals = np.linalg.eigvals(Astar)
print("Eigenvalues:", eigvals)

# ILR transform constant term
fstar = np.dot(np.log(f/gmean(f)), psi.T)

# Solve for the fixed point at t -> inf
fixedpoint = np.linalg.solve(Astar, -fstar)
# Inverse ILR transform the fixed point back to the simplex
equil_comp = np.exp(np.matmul(fixedpoint, psi))/sum(np.exp(np.matmul(fixedpoint, psi)))
print("Equilibrium composition:", equil_comp)

# The eigenvalues have both real and complex parts and the sum of them is less than zero,
# so the fixed point is a stable spiral, which means that the compositional process
# describes a damped harmonic oscillator.

# Now we go for the full solution. First we need the eigenvectors
eigenvectors = np.linalg.eig(Astar)[1]

# The general solution (in real-space) is x(t)=c1*exp(lam1*t)*v1 + c2*exp(lam2*t)*v2 + fixedpoint
# where lam are the eigenvalues, v are the eigenvectors and c are coefficients
# depending on the initial conditions. We let the c's be equal to 1.

time = np.arange(0, 10, 0.1)
x = [eigenvectors[0] * np.exp(eigvals[0]*t) + eigenvectors[1] *
     np.exp(eigvals[1]*t) + fixedpoint for t in time]

# Plot the ILR coordinates in time
plt.clf()
plt.plot(time, x)

# Plot the ILR coordinates against each other
plt.clf()
plt.plot([i[0] for i in x], [i[1] for i in x])

# Backtransform to the simplex and plot the compositional process
comp = pd.DataFrame(x).coda.ilr_inv(psi).coda.closure(1)

plt.clf()
plt.plot(time, comp[0])
plt.plot(time, comp[1])
plt.plot(time, comp[2])


print('\nExercise 9.3')

# Just run the script and plot he result
