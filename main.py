import numpy
from load_data import load_data
from nn import nn
from ova import ova
from pca import pca
from rfe import rfe
from tsne import tsne
# import json
# numpy.set_printoptions(threshold=1000000)

train_values, train_values_rfe, train_classes, train_classes_binary, test_values, test_values_rfe, test_classes, test_classes_binary, class_desc\
    = load_data()

all_values = numpy.concatenate((train_values, test_values))
all_classes = numpy.concatenate((train_classes, test_classes))
all_classes_binary = numpy.concatenate((train_classes_binary, test_classes_binary))

all_values_rfe = numpy.concatenate((train_values_rfe, test_values_rfe))

# print(all_values_rfe.shape)
# print(all_values_rfe)

# print(all_values_rfe[:,13])
# print(all_values[:,46])


# print(all_values)
# print(all_classes)

# t-sne on train data
# tsne(train_values, train_classes, class_desc, 0)

# t-sne on all data
# tsne(all_values, all_classes, class_desc, 0)

# pca
# pca = pca(train_values)
# print(pca)


# x = numpy.asarray(all_values, dtype=float)
# y = numpy.asarray(all_classes, dtype=int)

# rfe
# n_features, support, ranking, grid_scores, estimator = rfe(x,
#                                                            y)
# print(n_features, support, ranking, grid_scores, estimator)

# f = open('log.log', 'a')
# f.write(str(n_features))
# f.write(str(support))
# f.write(str(ranking))
# f.write(str(grid_scores))
# f.write(str(estimator))
# f.close()

# nn
result, history = nn(train_values_rfe, train_classes_binary, test_values_rfe, test_classes_binary)
print(result)
print(max(history.history['val_acc']))

# for perc in [100]:
# for perc in [100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5]:
    # print()
    # for deg in range(6):
    # deg = 1
    # print('test accuracy with deg ', deg, ', perc: ', perc, ' - ', ova(train_values_rfe, train_classes, test_values_rfe,
    #                                                                        test_classes, class_desc, deg=deg, perc=perc))
