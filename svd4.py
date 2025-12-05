#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 11:28:39 2025

@author: spike
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

# cargar imagen
img = imread("imagen.png")
img_gray = img.mean(axis=2)

U, S, Vt = np.linalg.svd(img_gray, full_matrices=False)

k = 50  # número de valores singulares
img_k = (U[:, :k] * S[:k]) @ Vt[:k, :]

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(img_gray, cmap='gray')
plt.title("Imagen original (gris)")

plt.subplot(1,2,2)
plt.imshow(img_k, cmap='gray')
plt.title("Reconstrucción con k=50")
plt.show()