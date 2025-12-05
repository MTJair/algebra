#! python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 10:27:40 2025

@author: spike
"""

import numpy as np

def clasificar_sistema(A, b):
    A = np.array(A, float)
    b = np.array(b, float)
    rA = np.linalg.matrix_rank(A)
    rAb = np.linalg.matrix_rank(np.column_stack([A, b]))

    print("rank(A) =", rA, " | rank([A|b]) =", rAb)

    n = A.shape[1]
    if rA == rAb == n:
        print("→ Solución única")
    elif rA == rAb < n:
        print("→ Infinitas soluciones")
    else:
        print("→ Sistema incompatible (sin solución)")

# Ejemplo
A = np.array([[1, -1], [2, -2]])
b = np.array([1, 3])
clasificar_sistema(A, b)
