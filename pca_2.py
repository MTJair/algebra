#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 11:19:05 2025

@author: spike
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3)
X = np.random.multivariate_normal([0,0], [[3,1],[1,1]], 300)

# centrar
Xc = X - X.mean(axis=0)

# covarianza
C = np.cov(Xc.T)

# autovalores y autovectores
eigvals, eigvecs = np.linalg.eig(C)

# componente principal
v1 = eigvecs[:, np.argmax(eigvals)]

# graficar datos y primer componente
plt.scatter(X[:,0], X[:,1], alpha=0.3)
plt.quiver(X.mean(axis=0)[0], X.mean(axis=0)[1],
           v1[0], v1[1], color='red', scale=3)
plt.title("Primer componente principal")
plt.grid(True)
plt.show()