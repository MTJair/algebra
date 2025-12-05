#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 11:23:47 2025

@author: spike
"""

import numpy as np
import matplotlib.pyplot as plt

# señal seno + ruido
t = np.linspace(0, 10, 2000)
signal = np.sin(2*np.pi*2*t)
noise = 0.3*np.random.randn(len(t))
x = signal + noise

# construir ventanas (embedding)
window = 30
X = np.array([x[i:i+window] for i in range(len(x)-window)])

# PCA en la matriz de ventanas
Xc = X - X.mean(axis=0)
C = np.cov(Xc.T)
eigvals, eigvecs = np.linalg.eig(C)

# reconstrucción usando primer componente
pc1 = eigvecs[:, np.argmax(eigvals)]
proj = Xc @ pc1

plt.plot(t[:-window], proj)
plt.title("Señal PCA (primer componente)")
plt.grid(True)
plt.show()