#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 11:27:09 2025

@author: spike
"""

import numpy as np
import matplotlib.pyplot as plt

# matriz
A = np.array([[2,1],
              [1,3]], float)

# SVD
U, S, Vt = np.linalg.svd(A)

# círculo unitario
theta = np.linspace(0, 2*np.pi, 200)
circle = np.vstack([np.cos(theta), np.sin(theta)])

# aplicar A
ellipse = A @ circle

plt.figure(figsize=(6,6))
plt.plot(circle[0], circle[1], label='Círculo unitario')
plt.plot(ellipse[0], ellipse[1], label='Imagen bajo A')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.title('Transformación geométrica de A')
plt.show()