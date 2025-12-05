#! python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 10:49:56 2025

@author: spike
"""

import numpy as np

def son_independientes(vectores):
    A = np.column_stack(vectores)
    rango = np.linalg.matrix_rank(A)
    return rango == len(vectores)

v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])  # múltiplo → dependiente
v3 = np.array([1, 0, 1])

print("Independencia:", son_independientes([v1, v2, v3]))

A = np.column_stack([v1, v2, v3])
U, S, Vt = np.linalg.svd(A)

# vectores con singular values ~ 0 → combinación de dependencia
nullspace = Vt[-1]
print("Combinación que da 0:", nullspace)
print("Verificación:", A @ nullspace)
