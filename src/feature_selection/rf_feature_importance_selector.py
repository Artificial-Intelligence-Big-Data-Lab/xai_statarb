from .feature_selector_base import *
import numpy as np


class RFFeatureImportanceSelector(FeatureSelectorBase):
    def __init__(self, k=1, selector: BaseFeatureSelector = None):
        selector = SelectKWorst(k=1, threshold=10000)
        super(RFFeatureImportanceSelector, self).__init__(k=k, selector=selector)

    def fit_transform(self, estimator, x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame,
                      y_test: pd.DataFrame):
        print('*' * 20, 'feature importance', '*' * 20)
        permutation_importance_s = self.__compute_feature_importance(x_test, y_test, estimator)
        for idx, min_row, columns, selection_error in self._selector.select_enumerate(x_test.columns,
                                                                                      permutation_importance_s):
            yield idx, min_row, columns, selection_error

    def __compute_feature_importance(self, x_test, y_test, estimator):
        all_columns = x_test.columns
        x_test_v = x_test.values
        y_test_v = y_test.values

        result = pd.DataFrame(
            {'features': all_columns, "feature_importance": estimator.feature_importances_})

        all_importance_values = [tree.feature_importances_ for i, tree in enumerate(estimator.estimators_)];
        res = pd.DataFrame(data=all_importance_values, columns=x_test.columns,
                           index=range(0, len(estimator.estimators_)))

        imp_ci = 1.96 * res.std(ddof=0) / np.sqrt(len(estimator.estimators_))
        permutation_importance = pd.DataFrame(
            dict(
                feature_importance=res.mean(),
                ci_fixed=imp_ci,
                errors=None,
                std_errors=None,
                success_count=None
            ),
        )

        return permutation_importance
