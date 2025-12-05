#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 11:28:01 2025

@author: spike
"""

import numpy as np

A = np.array([[4, 0],
              [3, -5]], float)

# 1) autovalores de A^T A
ATA = A.T @ A
eigvals, V = np.linalg.eig(ATA)

# valores singulares
S = np.sqrt(eigvals)

# U = A V / sigma
U = (A @ V) / S

print("A:\n", A)
print("\nValores singulares:", S)
print("\nV (autovectores de A^T A):\n", V)
print("\nU calculada manualmente:\n", U)