#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 10:57:08 2025

@author: spike
"""

import numpy as np

A = np.array([[1,2,3],
              [2,4,6],
              [1,0,1]], float)

# Usamos SVD para obtener base del espacio columna
U, S, Vt = np.linalg.svd(A)
rango = np.sum(S > 1e-10)
base_columna = U[:, :rango]

print("Rango:", rango)
print("Base del espacio columna:")
print(base_columna)