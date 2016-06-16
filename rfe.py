from sklearn.feature_selection import RFECV
from sklearn.svm import SVC


def rfe(data, labels):
    estimator = SVC(kernel="linear")
    selector = RFECV(estimator, verbose=1, scoring='accuracy')
    selector.fit(data, labels)
    return selector.n_features_, selector.support_, selector.ranking_, selector.grid_scores_, selector.estimator_
