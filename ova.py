from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import numpy
import matplotlib.pyplot as plt


def plot_confusion_matrix(desc, cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(16, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(desc))
    plt.xticks(tick_marks, desc, rotation=45)
    plt.yticks(tick_marks, desc)
    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def ova(train_values, train_classes, test_values, test_classes, desc, deg=2, perc=100):
    # scaler2 = preprocessing.MinMaxScaler((-1, 1)).fit(train_values)
    # train_values_scaled = scaler2.transform(train_values)
    # test_data_scaled = scaler2.transform(test_data)

    # scaler = preprocessing.StandardScaler().fit(train_values)
    # train_values_scaled = scaler.transform(train_values)
    # test_data_scaled = scaler.transform(test_data)

    all_values = numpy.concatenate((train_values, test_values))
    all_classes = numpy.concatenate((train_classes, test_classes))

    x = SelectPercentile(f_classif, percentile=perc).fit_transform(all_values, all_classes)

    svm = SVC(kernel='poly', degree=deg, random_state=0)

    ova = OneVsRestClassifier(svm)
    ova.fit(x[:144, :], train_classes)
    score = ova.score(x[144:, :], test_classes)

    cm = confusion_matrix(test_classes, ova.predict(x[144:, :]))
    print('Confusion matrix')
    print(cm)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)

    plot_confusion_matrix(desc, cm_normalized, title='Normalized confusion matrix')
    # score = ova.score(x[:144, :], train_classes)

    # ova.fit(train_values, train_classes)
    # score = ova.score(test_values, test_classes)

    # ova.fit(train_values_scaled, train_classes)
    # score = ova.score(test_data_scaled, test_classes)
    return score
