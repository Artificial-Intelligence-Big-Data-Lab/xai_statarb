import pandas as pd

from .feature_selector_base import FeatureSelectorBase


class RFFeatureImportanceSelector(FeatureSelectorBase):

    def fit_transform(self, estimator, x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame,
                      y_test: pd.DataFrame):
        print('*' * 20, 'feature importance', '*' * 20)
        all_columns = x_train.columns
        feat_imp_s = pd.DataFrame(
            {'features': all_columns, "feature_importance": estimator.feature_importances_}).sort_values(
            'feature_importance', ascending=False)
        column = feat_imp_s['features'].tail(self._k).values
        columns = set(all_columns) - set(column)
        self._importance = feat_imp_s[['features', 'feature_importance']].tail(self._k).values
        return columns