#! python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 10:33:34 2025

@author: spike
"""
import numpy as np

A = np.array([[1,1],[2,-1]], float)
b = np.array([3,0], float)

Q, R = np.linalg.qr(A)
x = np.linalg.solve(R, Q.T @ b)

print("Solución (QR):", x)
