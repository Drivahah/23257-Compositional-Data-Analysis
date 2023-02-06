'''
    Solutions to the exercises in chapter 2

'''

import numpy as np

# Exercise 2.2

# Values for kappa are:
# Percent: 100
# PPM: 1000000
# The unit for metagenomics is "read-depth", the number of sequenced reads


# Exercise 2.3

# Technically speaking, one cannot verify that data is compositional, but
# notice that each sample (column) sums up to 100, so it appears as if the
# data have been closed to that number. Real multivariate data are not
# constrained to a fixed sum. Also there are no negative numbers and no zeros.

# Exercise 2.4

s1 = [79.07, 20, 93]
s2 = [31.74, 68.26]

# Amalgamation do preserve closure since the compositions still sum to 100
# after amalgamation.

# Exercise 2.5

s1 = [79.07, 12.83, 8.10]
s2 = [31.74, 56.69, 11.57]

# Euclidean distance:

deuclid = np.sqrt(sum([pow(s1[i]-s2[i], 2) for i in [0, 1, 2]]))
dmanhattan = sum([np.abs(s1[i]-s2[i]) for i in [0, 1, 2]])

print(deuclid, dmanhattan)

# Close to 95

s1 = [95./sum(s1) * i for i in s1]
s2 = [95./sum(s2) * i for i in s2]

# Add a fourth part

s1 = s1 + [5]
s2 = s2 + [5]

# Recalculate Euclidean distance

deuclid = np.sqrt(sum([pow(s1[i]-s2[i], 2) for i in [0, 1, 2, 3]]))
dmanhattan = sum([np.abs(s1[i]-s2[i]) for i in [0, 1, 2]])
print(deuclid, dmanhattan)

# The distance is seen to decrease from 64.62 to 61.39, which shows that
# the Euclidean distance does not obey the principle of compositional
# coherence. Same for Manhattan (94.66 -> 89.93).
