from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn import preprocessing
import numpy


def ova(train_values, train_classes, test_values, test_classes, deg=2, perc=100):
    # scaler2 = preprocessing.MinMaxScaler((-1, 1)).fit(train_values)
    # train_values_scaled = scaler2.transform(train_values)
    # test_data_scaled = scaler2.transform(test_data)

    # scaler = preprocessing.StandardScaler().fit(train_values)
    # train_values_scaled = scaler.transform(train_values)
    # test_data_scaled = scaler.transform(test_data)

    all_values = numpy.concatenate((train_values, test_values))
    all_classes = numpy.concatenate((train_classes, test_classes))

    x = SelectPercentile(f_classif, percentile=perc).fit_transform(all_values, all_classes)
    # print(x.shape)
    print((x[:144, :]).shape)
    print((x[144:, :]).shape)

    svm = SVC(kernel='poly', degree=deg, random_state=0)
    ova = OneVsRestClassifier(svm)

    ova.fit(x[:144, :], train_classes)
    score = ova.score(x[144:, :], test_classes)

    # ova.fit(train_values, train_classes)
    # score = ova.score(test_values, test_classes)

    # ova.fit(train_values_scaled, train_classes)
    # score = ova.score(test_data_scaled, test_classes)
    return score
