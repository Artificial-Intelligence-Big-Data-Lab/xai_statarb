from abc import ABC


class FeatureSelectorBase(ABC):
    def __init__(self, k=1):
        self._columns = None
        self._importance = None
        self._k = k

    def transform(self, estimator, X_train, y_train, X_test, y_test):
        return self.fit_transform(estimator, X_train, y_train, X_test, y_test)

    def fit_transform(self, estimator, X, y, X_test, y_test):
        pass

    @property
    def importance(self):
        return self._importance