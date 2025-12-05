#! python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 10:57:25 2025

@author: spike
"""
import numpy as np

def nullspace(A, tol=1e-12):
    U, S, Vt = np.linalg.svd(A)
    r = np.sum(S > tol)
    return Vt[r:].T

A = np.array([[1,2,3],
              [2,4,6],
              [1,0,1]], float)
ns = nullspace(A)
print("Base del espacio nulo:")
print(ns)
