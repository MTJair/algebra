#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 10:40:45 2025

@author: spike
"""

import numpy as np

A = np.array([[4, 2],
              [2, 3]], float)
b = np.array([6, 5], float)

L = np.linalg.cholesky(A)

# resolver Ly = b
y = np.linalg.solve(L, b)

# resolver Lᵀx = y
x = np.linalg.solve(L.T, y)

print("Solución (Cholesky):", x)