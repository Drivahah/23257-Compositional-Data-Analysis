# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 19:32:48 2023

@author: pf259
"""

import numpy as np

class CompositionalVector():
    '''
    Parameters:
        s: list
        represents the parts of a composition of size len(x)
        
    Methods:
        closure: return a compositional vecture obtained applying closure with 
        value K to x with
        K: closure value
    '''
    def __init__(self, s):
        self.s = np.array(s)
    def closure(self, K):
        return K/self.s.sum()*self.s
    
class CompositionalData():
    '''
    Parameters:
        data: list of lists
    Methods:
        amalgamate: sum the samples specified with a list of indexes ind. 
            The new data is stored
    '''
    def __init__(self, data):
        self.data = np.array(data)
        self.data = np.transpose(self.data)
        
    def amalgamate(self, ind: list):
        self.data = np.vstack([self.data, [0]*len(self.data[0])])
        ind.sort(reverse=True)
        self.data[-1] = np.sum(self.data[ind, :], axis=0)
        self.data = np.delete(self.data, ind, axis=0)
    
    def printValues(self):
        for i in range(len(self.data)):
            print('x' + str(i+1) + ': ' + str(self.data[i]))
        print('Closure: ' + str(np.sum(self.data, axis=0)))
    
    def euclidianDistance(self, ind):
        v1 = self.data[ind[0]]
        v2 = self.data[ind[1]]
        squaredDiff = np.square(np.subtract(v1, v2))
        return np.sqrt(np.sum(squaredDiff))
    
    def manhattanDistance(self, ind):
        v1 = self.data[ind[0]]
        v2 = self.data[ind[1]]
        absDiff = np.absolute(np.subtract(v1, v2))
        return np.sum(absDiff)
        
    
    def appendDimension(self, x):
        self.data = np.vstack([self.data, x])
    
    def closure(self, K):
        self.data = K / np.sum(self.data, axis=0) * self.data
        