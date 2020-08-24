#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:03:10 2019

@author: negar
"""
n = 3
a1 = np.zeros((3,3))
b1 = np.ones((3,3))
k = 0
nn = [1,2,3,4,5,6]
for i in range(3):
    for j in range(i,3):
        a1[i][j] = b1[i][j]+nn[k]
        a1[j][i] = a1[i][j]
        k = k+1
