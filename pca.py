import numpy as np
from sklearn.decomposition import PCA


def pca(data):
    pca = PCA()
    pca.fit(data)
    return pca.explained_variance_ratio_

