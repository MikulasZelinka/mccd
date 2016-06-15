import numpy
from load_data import load_data
from nn import nn
from ova import ova
from pca import pca
from rfe import rfe
from tsne import tsne

train_values, train_classes, train_classes_binary, test_values, test_classes, test_classes_binary, class_desc\
    = load_data()

all_values = numpy.concatenate((train_values, test_values))
all_classes = numpy.concatenate((train_classes, test_classes))
all_classes_binary = numpy.concatenate((train_classes_binary, test_classes_binary))

# t-sne on train data
# tsne(train_values, train_classes, class_desc, 0)

# t-sne on all data
# tsne(all_values, all_classes, class_desc, 0)

# pca
# pca = pca(train_values)
# print(pca)

# rfe
# n_features, support, ranking, grid_scores, estimator = rfe(numpy.asarray(all_values, dtype=float), numpy.asarray(all_classes, dtype=int))
# print(n_features, support, ranking, grid_scores, estimator)

# nn
# result, history = nn(train_values, train_classes_binary, test_values, test_classes_binary)
# print(result)
#
# print(max(history.history['val_acc']))

print('test accuracy', ova(train_values, train_classes, test_values, test_classes))
