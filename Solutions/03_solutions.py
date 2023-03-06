import matplotlib.pyplot as plt
import numpy as np

print('\nExercise 3.1')

x = [0.7, 0.5, 0.8]
y = [0.25, 0.75, 0.5]

# Without closing before perturbing:
xy = [x[i]*y[i] for i in range(3)]
xypercent = [100./sum(xy) * xy[i] for i in range(3)]
print(xypercent)
# array([18.4, 39.5, 42.1])

# With closure to 100 before perturbing:
xpercent = [100./sum(x) * x[i] for i in range(3)]
ypercent = [100./sum(y) * y[i] for i in range(3)]
xy = [xpercent[i]*ypercent[i] for i in range(3)]
xypercent = [100./sum(xy) * xy[i] for i in range(3)]

print(xypercent)
# array([18.4, 39.5, 42.1])

# The result is the same, whether or not you close before perturbing.


print('\nExercise 3.2')

x = [0.7, 0.4, 0.8]
y = [2., 8., 1.]

aip = 1./(2*3) * sum([np.log(x[i]/x[j])*np.log(y[i]/y[j])
                      for i in range(3) for j in range(3)])

print(aip)
# -0.77; not orthogonal. Orthogonal vectors have a dot product equal to 0


print('\nExercise 3.3')

x = [3.74, 9.35, 16.82, 18.69, 23.36, 28.04]
y = [9.35, 28.04, 16.82, 3.74, 18.69, 23.36]

# Aitchison norm
an_x = np.sqrt(1./(2*len(x)) *
               sum([sum([pow(np.log(i/j), 2) for i in x]) for j in x]))
an_y = np.sqrt(1./(2*len(x)) *
               sum([sum([pow(np.log(i/j), 2) for i in y]) for j in y]))
# Aichison inner product
aip = 1./(2*len(x)) * sum([sum([np.log(x[i]/x[j])*np.log(y[i]/y[j])
                                for i in range(6)]) for j in range(6)])

angle = np.arccos(aip)/(an_x*an_y)

print(angle)
# 0.42


print('\nExercise 3.4')

x = [0.7, 0.4, 0.8]
a = np.sqrt(1./(2.*3) * sum([np.log(x[i]/x[j]) **
            2 for i in range(3) for j in range(3)]))
alpha = 1./a
alphadotx = [pow(x[i], alpha) for i in range(3)]
anorm = np.sqrt(1./(2.*3) * sum([np.log(alphadotx[i]/alphadotx[j])
                                 ** 2 for i in range(3) for j in range(3)]))

print(anorm)
# 1.; We have performed a generalized multiplication of a composition and its
# inverse norm, that is, we have (generalized) divided the composition with its
# length, which is to normalize the vector. The length (norm) of a normalized vector
# is unity.


print('\nExercise 3.5')
# We use the first to compositions from Exercise 2.3:

x1 = [79.07, 12.83, 8.10]
x2 = [31.74, 56.69, 11.57]

ad = np.sqrt(1./(2.*3) * sum([(np.log(x1[i]/x1[j])-np.log(x2[i]/x2[j])) ** 2
                              for i in range(3) for j in range(3)]))
print(ad)
# 1.697

# Close to 95 and add fourth part
x1_95 = [95./sum(x1) * x1[i] for i in range(3)]
x2_95 = [95./sum(x2) * x2[i] for i in range(3)]

x1 = x1_95+[5]
x2 = x2_95+[5]

ad = np.sqrt(1./(2.*4) * sum([(np.log(x1[i]/x1[j])-np.log(x2[i]/x2[j])) ** 2
                              for i in range(4) for j in range(4)]))
print(ad)
# 1.718

# 1.718 > 1.697, that is the distance is greater when a fourth part is added
# Aichison distance obey the sub-compositional coherence principle

print('\nExercise 3.6')

# Use data from Exercise 2.3

data = np.array([[79.07, 12.83, 8.10], [31.74, 56.69, 11.57],
                 [18.61, 72.05, 9.34], [49.51, 15.11, 35.38],
                 [29.22, 52.36, 18.42]])

# First calculate the geometric means of the compositions
gm = [pow(data[i][0]*data[i][1]*data[i][2], 1./3) for i in range(5)]
clr = [[np.log(data[i][j]/gm[i]) for j in range(3)] for i in range(5)]
print(np.round(clr, 2))
print(np.round([sum(clr[i]) for i in range(5)], 2))


print('\nExercise 3.7')

# Use data from Exercise 2.3

data = np.array([[79.07, 12.83, 8.10], [31.74, 56.69, 11.57],
                 [18.61, 72.05, 9.34], [49.51, 15.11, 35.38],
                 [29.22, 52.36, 18.42]])

# use the third part as denominator for an ALR tranformation
alr = [[np.log(data[i][0]/data[i][2]), np.log(data[i][1]/data[i][2])]
       for i in range(5)]
_ = [plt.plot([alr[i][0]], [alr[i][1]], 'o', color='maroon') for i in range(5)]

# now use the first part as denominator for an ALR tranformation
alr = [[np.log(data[i][1]/data[i][0]), np.log(data[i][2]/data[i][0])]
       for i in range(5)]
_ = [plt.plot([alr[i][0]], [alr[i][1]], 'o', color='steelblue')
     for i in range(5)]

# and finally the second part as denominator for an ALR tranformation
alr = [[np.log(data[i][0]/data[i][1]), np.log(data[i][2]/data[i][1])]
       for i in range(5)]
_ = [plt.plot([alr[i][0]], [alr[i][1]], 'o', color='seagreen')
     for i in range(5)]

plt.show()

# The resulting coordinates in real space are not the same

print('\nExercise 3.8')

# Two vectors are not orthogonal if the dot product differs from 0
# Calculate the Aichison inner product between any two vectors in the basis

# For D=3 as example
x1 = [np.e, 1, 1]
x2 = [1, np.e, 1]

aip = 1./(2*3) * sum([sum([np.log(x1[i]/x1[j])*np.log(x2[i]/x2[j])
                           for i in range(3)]) for j in range(3)])

print(aip)
# -0.33, not zero


print('\nExercise 3.9')
# First we build a binary partition basis
ptb = [[1, -1, -1], [0, 1, -1]]

# Then we nomalize it using formula 3.22 in the lecture notes:
ptbn = [[1./1*np.sqrt(2./3), -1./2*np.sqrt(2./3), -1./2*np.sqrt(2./3)],
        [0, 1./1*np.sqrt(1./2), -1./1*np.sqrt(1./2)]]

print(ptbn)


print('\nExercise 3.10')
# Use data from Exercise 2.3

data = np.array([[79.07, 12.83, 8.10], [31.74, 56.69, 11.57],
                 [18.61, 72.05, 9.34], [49.51, 15.11, 35.38],
                 [29.22, 52.36, 18.42]])

gm = [pow(data[i][0]*data[i][1]*data[i][2], 1./3) for i in range(5)]
clr = [[np.log(data[i][j]/gm[i]) for j in range(3)] for i in range(5)]

# Use ptbn from previous exercise
ilr = [np.dot(np.array(clr[i]), np.array(ptbn).T) for i in range(5)]
_ = [plt.plot([ilr[i][0]], [ilr[i][1]], 'x', color='purple') for i in range(5)]

# Using another binary partion basis
ptb = [[1, 1, -1], [1, -1, 0]]

# Then we nomalize it using formula 3.22 in the lecture notes:
ptbn = [[1./2*np.sqrt(2./3), 1./2*np.sqrt(2./3), -1./1*np.sqrt(2./3)],
        [1./1*np.sqrt(1./2), -1./1*np.sqrt(1./2), 0]]

ilr = [np.dot(np.array(clr[i]), np.array(ptbn).T) for i in range(5)]
_ = [plt.plot([ilr[i][0]], [ilr[i][1]], 'x', color='salmon') for i in range(5)]

# Notice that the two sets of ILR coordinates are equally interspaced, just flipped on both axis
# The ALR coordinates does not have this quality.
