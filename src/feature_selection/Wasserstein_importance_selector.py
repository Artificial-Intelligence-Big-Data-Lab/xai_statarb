import numpy as np

from . import PISelector
from .feature_selector_base import *
from .wasserstein_distance import wasserstein_distance_from_samples


class WassersteinFeatureImportanceSelector(PISelector):

    def __init__(self, k=0, num_rounds=50, seed=0):
        super(WassersteinFeatureImportanceSelector, self).__init__(k=k, num_rounds=num_rounds
                                                                   , metric=wasserstein_distance_from_samples, seed=seed)
        self.name = "pi_ws_{0}".format("all" if k == 0 else k)

    def _feature_importance_permutation(self, estimator_fn, x_test, y_test):

        rng = np.random.RandomState(self._seed)
        x_test_v = x_test.values
        y_test_v = y_test.values
        mean_importance_values = np.zeros(x_test_v.shape[1])
        all_importance_values = np.zeros((x_test_v.shape[1], self._num_rounds))
        all_error_values = np.zeros((x_test_v.shape[1], self._num_rounds))

        for round_idx in range(self._num_rounds):
            for col_idx in range(x_test_v.shape[1]):
                save_col = x_test_v[:, col_idx].copy()
                rng.shuffle(x_test_v[:, col_idx])
                new_score = self._loss(y_test_v, estimator_fn(x_test_v))
                x_test_v[:, col_idx] = save_col
                if self._loss.__name__ in list(['r2', 'accuracy']):
                    importance = (self._original_loss - new_score)
                else:
                    importance = (new_score - self._original_loss)
                mean_importance_values[col_idx] += importance
                all_importance_values[col_idx, round_idx] = importance
                all_error_values[col_idx, round_idx] = new_score

        all_feat_imp_df = pd.DataFrame(data=np.transpose(all_importance_values), columns=x_test.columns,
                                       index=range(0, self._num_rounds))
        all_trial_errors_df = pd.DataFrame(data=np.transpose(all_error_values), columns=x_test.columns,
                                           index=range(0, self._num_rounds))
        return all_feat_imp_df, all_trial_errors_df
