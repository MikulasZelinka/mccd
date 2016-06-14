from sklearn.feature_selection import RFECV
from sklearn.svm import SVR


def rfe(data, labels):

    estimator = SVR(kernel="linear")
    selector = RFECV(estimator, verbose=2)
    selector = selector.fit(data, labels)
    return selector.n_features, selector.support_, selector.ranking_, selector.grid_scores_, selector.estimator_
