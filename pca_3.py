#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 11:20:44 2025

@author: spike
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

# cargar imagen
img = imread("imagen.png")
img = img.mean(axis=2)   # gris

# aplicar PCA usando SVD (equivalente numérico)
U, S, Vt = np.linalg.svd(img, full_matrices=False)

# reconstruir usando k componentes
k = 50
img_k = (U[:, :k] * S[:k]) @ Vt[:k, :]

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(img_k, cmap='gray')
plt.title(f"Reconstrucción con k={k}")
plt.show()