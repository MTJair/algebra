#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 11:25:46 2025

@author: spike
"""

import numpy as np

A = np.array([[3, 1],
              [1, 3],
              [0, 2]], float)

# valores singulares
U, S, Vt = np.linalg.svd(A)

print("Valores singulares:", S)