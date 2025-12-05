#! python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 10:24:22 2025

@author: spike
"""
import numpy as np

def cramer(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    detA = np.linalg.det(A)
    if abs(detA) < 1e-12:
        raise ValueError("El sistema no tiene solución única (det(A)=0)")
    n = A.shape[1]
    x = np.zeros(n)
    for i in range(n):
        Ai = A.copy()
        Ai[:, i] = b
        x[i] = np.linalg.det(Ai) / detA
    return x

# Ejemplo
A = np.array([[1, 2],
              [3, -1]])
b = np.array([5, 4])

sol = cramer(A, b)
print("Solución por Cramer:", sol)
