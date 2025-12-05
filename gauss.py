#! python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 10:28:36 2025

@author: spike
"""

import numpy as np

def gauss(A, b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    m, n = A.shape

    # Matriz aumentada
    M = np.hstack([A, b.reshape(-1,1)])

    for i in range(n):
        # 1. Seleccionar pivote (pivoteo parcial)
        pivot = i + np.argmax(np.abs(M[i:, i]))
        if abs(M[pivot, i]) < 1e-12:
            continue  # columna sin pivote

        # Intercambio de filas
        M[[i, pivot]] = M[[pivot, i]]

        # 2. Normalizar pivote
        M[i] = M[i] / M[i, i]

        # 3. Eliminar hacia abajo
        for j in range(i+1, m):
            M[j] -= M[j, i] * M[i]

    return M

# Ejemplo
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]], float)
b = np.array([8, -11, -3], float)

M = gauss(A, b)
print("Matriz escalonada:")
print(M)

def back_substitution(M):
    M = np.array(M, float)
    m, n1 = M.shape
    n = n1 - 1

    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = M[i, -1] - np.dot(M[i, i+1:n], x[i+1:n])
    return x

# Solución del sistema
x = back_substitution(M)
print("Solución:", x)
