#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 11:30:08 2025

@author: spike
"""

import numpy as np

# datos sintéticos de 100 muestras y 10 variables
np.random.seed(1)
X = np.random.randn(100, 10)

# centrar
Xc = X - X.mean(axis=0)

# SVD
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

# proyectar sobre los primeros k componentes
k = 3
X_k = Xc @ Vt[:k].T

print("Dimensiones originales:", X.shape)
print("Nuevas dimensiones reducidas:", X_k.shape)