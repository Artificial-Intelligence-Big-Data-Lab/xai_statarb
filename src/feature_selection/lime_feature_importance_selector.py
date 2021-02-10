import lime
import lime.lime_tabular
import pandas as pd
from lime import submodular_pick

from .feature_selector_base import FeatureSelectorBase


class LIMEFeatureImportanceSelector(FeatureSelectorBase):

    def __init__(self, k, num_exp_desired=126, random_state=123):
        super(LIMEFeatureImportanceSelector, self).__init__(k)
        self.__num_exp_desired = num_exp_desired
        self.__random_state = random_state

    def fit_transform(self, estimator, x_train, y_train, x_test, y_test):
        # ####***************LIME feature importance***********************
        print('*' * 20, 'LIME feature importance', '*' * 20)
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(x_train.values,
                                                                training_labels=y_train.values,
                                                                feature_names=x_train.columns.tolist(),
                                                                verbose=False, mode='regression'
                                                                , discretize_continuous=False
                                                                , random_state=self.__random_state)
        sp_obj_cr = submodular_pick.SubmodularPick(lime_explainer, x_test.values, estimator.predict,
                                                   num_features=len(x_test.columns),
                                                   num_exps_desired=self.__num_exp_desired)
        W_s = pd.DataFrame([dict(this.as_list(label=0)) for this in sp_obj_cr.explanations])
        rank_w_s = W_s[x_test.columns].abs().rank(1, ascending=False, method='first')
        rank_w_s_median, rank_w_s_mean = rank_w_s.median(), rank_w_s.mean()
        rank_w_s_median.name = 'median_rank'
        rank_w_s_mean.name = 'mean_rank'
        ranked_features = pd.concat([rank_w_s_median, rank_w_s_mean], axis=1).sort_values(
            by=['median_rank', 'mean_rank'],
            ascending=[False, False])
        min_row = ranked_features.index[:self._k].values
        columns = set(x_test.columns) - set(min_row)
        self._importance = ranked_features.head(self._k).reset_index().values
        return columns
