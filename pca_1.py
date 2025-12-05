#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 11:06:38 2025

@author: spike
"""

import numpy as np

A = np.array([[2, 1],
              [1, 2]], float)

eigvals, eigvecs = np.linalg.eig(A)
print("Autovalores:", eigvals)
print("Autovectores:\n", eigvecs)

for i in range(len(eigvals)):
    print("Chequeo", i, ":", A @ eigvecs[:,i], "≈", eigvals[i] * eigvecs[:,i])
    
A = np.array([[4,1],
              [1,4]], float)

eigvals, P = np.linalg.eig(A)
D = np.diag(eigvals)

print("Matriz diagonal D:\n", D)
print("Verificación P D P^{-1}:\n", P @ D @ np.linalg.inv(P))

A = np.array([[3,1],
              [1,2]], float)

# forma cuadrática Q(x)
def Q(x):
    return x.T @ A @ x

x = np.array([1,2], float)
print("Q(x) =", Q(x))


theta = np.pi/4
P = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

print("P^T P:\n", P.T @ P)