#! python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 10:47:47 2025

@author: spike
"""

import numpy as np

# vectores base
v1 = np.array([1, 2])
v2 = np.array([2, -1])

# coeficientes
a, b = 3, -2

combo = a * v1 + b * v2
print("Combinación lineal:", combo)
