from load_data import load_data
from tsne import tsne
from pca import pca
import numpy

train_values, train_classes, train_classes_binary, test_values, test_classes, test_classes_binary, class_desc\
    = load_data()

all_values = numpy.concatenate((train_values, test_values))
all_classes = numpy.concatenate((train_classes, test_classes))
all_classes_binary = numpy.concatenate((train_classes_binary, test_classes_binary))

# t-sne on train data
tsne(all_values, all_classes, class_desc, 0)

# pca
pca = pca(train_values)