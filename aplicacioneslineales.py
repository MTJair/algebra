#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 10:58:23 2025

@author: spike
"""

import numpy as np

# T(x,y) = (2x + y, -x + 3y)
A = np.array([[2, 1],
              [-1, 3]], float)

x = np.array([4, -2])
print("T(x) = A x =", A @ x)



def matriz_de_T(vectores_imagen):
    # vectores_imagen es una lista [T(e1), T(e2), ...]
    return np.column_stack(vectores_imagen)

T_e1 = np.array([2, -1])
T_e2 = np.array([1, 3])

A = matriz_de_T([T_e1, T_e2])
print("Matriz de T:\n", A)



A = np.array([[1,2,3],
              [2,4,6],
              [1,0,1]], float)

# rango (dim imagen)
rango = np.linalg.matrix_rank(A)

# nulidad (dim núcleo)
nulidad = A.shape[1] - rango

print("Rango =", rango)
print("Nulidad =", nulidad)


def nullspace(A, tol=1e-12):
    U, S, Vt = np.linalg.svd(A)
    r = np.sum(S > tol)
    return Vt[r:].T

ns = nullspace(A)
print("Base del núcleo:\n", ns)



def bases_de_A(A):
    # base del espacio columna
    U, S, Vt = np.linalg.svd(A)
    r = np.sum(S > 1e-12)
    base_col = U[:, :r]

    # base del núcleo
    base_null = Vt[r:].T
    return base_col, base_null

base_col, base_null = bases_de_A(A)
print("Base del espacio columna:\n", base_col)
print("Base del núcleo:\n", base_null)
print("Verificación:", base_col.shape[1] + base_null.shape[1], "=", A.shape[1])



# matriz de T en base estándar
A = np.array([[3, 1],
              [0, 2]])

# nueva base B = {b1, b2}
b1 = np.array([1, 1])
b2 = np.array([1, -1])
P = np.column_stack([b1, b2])   # matriz cambio de base

# matriz de T en la base B
A_B = np.linalg.inv(P) @ A @ P
print("Matriz de T en base B:\n", A_B)

# verificar que A_B es similar a A
A_recuperada = P @ A_B @ np.linalg.inv(P)
print("Recuperación de A:\n", A_recuperada)


import numpy as np

A = np.random.randn(3,3)
P = np.random.randn(3,3)

# asegurar invertibilidad
while np.linalg.det(P) == 0:
    P = np.random.randn(3,3)

B = np.linalg.inv(P) @ A @ P

# eigenvalores
eig_A = np.linalg.eigvals(A)
eig_B = np.linalg.eigvals(B)

print("Autovalores de A:", eig_A)
print("Autovalores de B:", eig_B)