from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


def ova(train_data, train_classes, test_data, test_classes):
    svm = SVC(kernel='linear')
    ova = OneVsRestClassifier(svm)
    ova.fit(train_data, train_classes)
    score = ova.score(test_data, test_classes)
    return score
