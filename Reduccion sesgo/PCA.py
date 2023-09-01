import csv

import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from numpy import loadtxt
matrix = []
with open('diferencia_pares_pre.txt') as f:
    for line in f:
        #inner_list = [elt.strip() for elt in line.split('	')]
        # in alternative, if you need to use the file content as numbers
        inner_list = [float(elt.strip()) for elt in line.split('	')]
        matrix.append(inner_list)

# Reduce dimensionality with PCA
pca = PCA(n_components=1)

# Fit and transform data
pc1=pca.fit_transform(matrix)

print(pc1)

np.savetxt('PCA_simple_pre.txt', pc1, delimiter=' ')