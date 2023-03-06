# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 19:33:17 2023

@author: pf259
"""

from compositional import CompositionalData

s1 = ([79.07, 12.83, 8.10])    
s2 = ([31.74, 56.69, 11.57])    
s3 = ([18.61, 72.05, 9.34])    
s4 = ([49.51, 15.11, 35.38])    
s5 = ([29.22, 52.36, 18.42])    
samples = CompositionalData([s1, s2, s3, s4, s5])
print('Exercise 2.3')
samples.printValues()
samples.amalgamate([1,2])
print('_____________________________________\n')
print('Exercise 2.4')
samples.printValues()
print('_____________________________________\n')
print('Exercise 2.5')
print('Euclidian distance between samples 1 and 2 (3D) ' + str(samples.euclidianDistance([0,1])))
print('Manhattan distance between samples 1 and 2 (3D) ' + str(samples.manhattanDistance([0,1])))
samples_closure = samples
samples_closure.closure(95)
samples_closure.appendDimension([5] * len(samples_closure.data[0]))
print('Euclidian distance between samples 1 and 2 (4D) ' + str(samples_closure.euclidianDistance([0,1])))
print('Manhattan distance between samples 1 and 2 (4D) ' + str(samples_closure.manhattanDistance([0,1])))
