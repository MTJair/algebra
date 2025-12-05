#! python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 10:48:56 2025

@author: spike
"""
import numpy as np

def es_combinacion_lineal(vectores, objetivo):
    A = np.column_stack(vectores)
    objetivo = np.array(objetivo)
    x, *_ = np.linalg.lstsq(A, objetivo, rcond=None)
    return A @ x, x  # aproximación y coeficientes

v1 = np.array([1, 1, 0])
v2 = np.array([0, 1, 1])
v3 = np.array([1, 0, 1])
objetivo = np.array([2, 2, 1])

aprox, coef = es_combinacion_lineal([v1, v2, v3], objetivo)
print("Vector generado:", aprox)
print("Coeficientes:", coef)
