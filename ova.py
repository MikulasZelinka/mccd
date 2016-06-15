from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn import preprocessing


def ova(train_data, train_classes, test_data, test_classes, deg=2, rs=0):
    # scaler2 = preprocessing.MinMaxScaler((-1, 1)).fit(train_data)
    # train_data_scaled = scaler2.transform(train_data)
    # test_data_scaled = scaler2.transform(test_data)

    # scaler = preprocessing.StandardScaler().fit(train_data)
    # train_data_scaled = scaler.transform(train_data)
    # test_data_scaled = scaler.transform(test_data)

    svm = SVC(kernel='poly', degree=deg, random_state=rs)
    ova = OneVsRestClassifier(svm)
    ova.fit(train_data, train_classes)
    score = ova.score(test_data, test_classes)
    # ova.fit(train_data_scaled, train_classes)
    # score = ova.score(test_data_scaled, test_classes)
    return score
