import lime
import lime.lime_tabular
import numpy as np
from lime import submodular_pick

from .feature_selector_base import *


class LIMEFeatureImportanceSelector(FeatureSelectorBase):

    def __init__(self, k, num_rounds=126, selector: BaseFeatureSelector = None, seed=123):
        self._seed = seed

        if not isinstance(num_rounds, int):
            raise ValueError('num_rounds must be an integer.')
        if num_rounds < 1:
            raise ValueError('num_rounds must be greater than 1.')

        self._num_rounds = num_rounds
        v_selector = SelectKWorst(1, threshold=1000) if selector is None else selector
        super(LIMEFeatureImportanceSelector, self).__init__(k=1, selector=v_selector)

    def fit_transform(self, estimator, x_train, y_train, x_test, y_test):
        # ####***************LIME feature importance***********************
        print('*' * 20, 'LIME feature importance', '*' * 20)
        permutation_importance_s = self.__compute_lime_importance(x_train, y_train, x_test, y_test, estimator)
        for idx, min_row, columns, selection_error in self._selector.select_enumerate(x_test.columns,
                                                                                      permutation_importance_s):
            yield idx, min_row, columns, selection_error

    def __compute_lime_importance(self, x_train, y_train, x_test,y_test,estimator):

        lime_explainer = lime.lime_tabular.LimeTabularExplainer(x_train.values,
                                                                training_labels=y_train.values,
                                                                feature_names=x_train.columns.tolist(),
                                                                verbose=False, mode='regression'
                                                                , discretize_continuous=False
                                                                , random_state=self._seed)
        sp_obj_cr = submodular_pick.SubmodularPick(lime_explainer, x_test.values, estimator.predict,
                                                   num_features=len(x_test.columns),
                                                   num_exps_desired=self._num_rounds)
        # W_s = pd.DataFrame([dict(this.as_list(label=0)) for this in sp_obj_cr.explanations])
        # rank_w_s = W_s[x_test.columns].abs().rank(1, ascending=False, method='first')
        # rank_w_s_median, rank_w_s_mean = rank_w_s.median(), rank_w_s.mean()
        # rank_w_s_median.name = 'median_rank'
        # rank_w_s_mean.name = 'mean_rank'
        # ranked_features = pd.concat([rank_w_s_median, rank_w_s_mean], axis=1).sort_values(
        #     by=['median_rank', 'mean_rank'],
        #     ascending=[False, False])
        # # min_row = ranked_features.index[:self._k].values
        # # columns = set(x_test.columns) - set(min_row)
        # # self._importance = ranked_features.head(self._k).reset_index().values
        res = pd.DataFrame([dict(this.as_list(label=0)) for this in sp_obj_cr.explanations])
        res = res.abs()
        imp_ci = 1.96 * res.std(ddof=0) / np.sqrt(len(res))
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
