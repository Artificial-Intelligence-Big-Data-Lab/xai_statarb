import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .feature_selector_base import *
import numpy as np


class PISelectorBase(FeatureSelectorBase):

    def __init__(self, k=0, num_rounds=50, selector: BaseFeatureSelector = None, metric=mean_squared_error, seed=0):
        """
        Parameters
        ----------
        k : number of features to be removed
        num_rounds : number of trials for permutation importance, default 50
            metric : str, callable
            The metric for evaluating the feature importance through
            permutation. By default, the strings 'accuracy' is
            recommended for classifiers and the string 'r2' is
            recommended for regressors. Optionally, a custom
            scoring function (e.g., `metric=scoring_func`) that
            accepts two arguments, y_true and y_pred, which have
            similar shape to the `y` array.
        seed : seed for random shuffling
        """
        super().__init__(k, selector)
        self._seed = seed

        if not isinstance(num_rounds, int):
            raise ValueError('num_rounds must be an integer.')
        if num_rounds < 1:
            raise ValueError('num_rounds must be greater than 1.')

        self._num_rounds = num_rounds

        if not (metric in ('r2', 'accuracy') or hasattr(metric, '__call__')):
            raise ValueError('metric must be either "r2", "accuracy", '
                             'or a function with signature func(y_true, y_pred).')

        if metric == 'r2':
            def score_func(y_true, y_pred):
                sum_of_squares = np.sum(np.square(y_true - y_pred))
                res_sum_of_squares = np.sum(np.square(y_true - y_true.mean()))
                r2_score = 1. - (sum_of_squares / res_sum_of_squares)
                return r2_score

        elif metric == 'accuracy':
            def score_func(y_true, y_pred):
                return np.mean(y_true == y_pred)

        else:
            score_func = metric

        self._loss = score_func

    def fit_transform(self, estimator, x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame,
                      y_test: pd.DataFrame):
        print('*' * 20, 'permutation importance', '*' * 20)
        self._original_loss = self._loss(y_test, estimator.predict(x_test))
        permutation_importance_s = self.__compute_permutation_importance(x_test, y_test, estimator)

        for idx, min_row, columns, selection_error in self._selector.select_enumerate(x_test.columns,
                                                                                      permutation_importance_s):
            yield idx, min_row, columns, selection_error

    def __compute_permutation_importance(self, x_test, y_test, estimator):
        res, res_error = self._feature_importance_permutation(estimator_fn=estimator.predict, x_test=x_test,
                                                              y_test=y_test['label'])

        count_check = res_error.applymap(lambda x: 1 if x <= self._original_loss else 0)
        imp_ci = 1.96 * res.std(ddof=0) / np.sqrt(self._num_rounds)
        permutation_importance = pd.DataFrame(
            dict(
                feature_importance=res.mean(),
                ci_fixed=imp_ci,
                errors=res_error.mean(),
                std_errors=res_error.std(),
                success_count=count_check.sum()
            ),
        )

        return permutation_importance


class PISelector(PISelectorBase):

    def __init__(self, k=0, num_rounds=50, metric=mean_squared_error, seed=0):
        """
        Parameters
        ----------
        k : number of features to be removed
        num_rounds : number of trials for permutation importance, default 50
            metric : str, callable
            The metric for evaluating the feature importance through
            permutation. By default, the strings 'accuracy' is
            recommended for classifiers and the string 'r2' is
            recommended for regressors. Optionally, a custom
            scoring function (e.g., `metric=scoring_func`) that
            accepts two arguments, y_true and y_pred, which have
            similar shape to the `y` array.
        seed : seed for random shuffling
        """
        selector = AllBellowZeroSelector() if k == 0 else SelectKWorst(k)
        super().__init__(k, num_rounds, selector, metric, seed)

        self.name = "pi2_{0}".format("all" if k == 0 else k)

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
                    importance = (self._original_loss - new_score) / self._original_loss
                else:
                    importance = (new_score - self._original_loss) / self._original_loss
                mean_importance_values[col_idx] += importance
                all_importance_values[col_idx, round_idx] = importance
                all_error_values[col_idx, round_idx] = new_score

        all_feat_imp_df = pd.DataFrame(data=np.transpose(all_importance_values), columns=x_test.columns,
                                       index=range(0, self._num_rounds))
        all_trial_errors_df = pd.DataFrame(data=np.transpose(all_error_values), columns=x_test.columns,
                                           index=range(0, self._num_rounds))
        return all_feat_imp_df, all_trial_errors_df


class PISelectorUnormalized(PISelector):

    def __init__(self, k=0, num_rounds=50, metric=mean_squared_error, seed=0):
        super(PISelectorUnormalized, self).__init__(k=k, num_rounds=num_rounds
                                                    , metric=metric, seed=seed)
        self.name = "pi3_{0}".format("all" if k == 0 else k)

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


class PISelectorKBest(PISelector):
    def __init__(self, num_rounds=50, metric=mean_squared_error, seed=42):
        super(PISelectorKBest, self).__init__(k=1, num_rounds=num_rounds, metric=metric, seed=seed)
        self._selector = SelectKBest(k=1)
        self.name = "pi3_1"


class PermutationImportanceSelector(PISelectorBase):

    def __init__(self, k=0, num_rounds=50, loss_fn=mean_absolute_error, seed=0):
        selector = AllBellowZeroSelector() if k == 0 else SelectKWorst(k)
        super().__init__(k, num_rounds, selector, loss_fn, seed)
        self.name = "pi_{0}".format("all" if k == 0 else k)

    @staticmethod
    def __pointwise_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        :param y_true: Array of shape (N,), (N, 1)
        :param y_pred: Array of shape (N,), (N, 1)

        :returns: An array of shape (N,) corresponding to the squared error of each909090p5r
            row.

        Parameters
        ----------
        self

        """
        # demote (N, 1) shapes to (N,)
        if y_true.ndim == 2:
            y_true = y_true[:, 0]
        if y_pred.ndim == 2:
            y_pred = y_pred[:, 0]

        loss = np.abs(y_true - y_pred)
        return loss

    @staticmethod
    def __permutation_predictions(model_fn, x, seed):
        """Generates a DataFrame where each column is the predictions resulting
        from permutation-based ablation of the corresponding feature."""
        shuffled_order = np.random.RandomState(seed).permutation(x.shape[0])
        res = dict()
        for column_name, column in x.iteritems():
            permuted_x = x.copy()
            permuted_x[column_name] = column.values[shuffled_order]
            res[column_name] = model_fn(permuted_x)
        return pd.DataFrame(res)

    def _feature_importance_permutation(self, estimator_fn, x_test, y_test):
        """Reference implementation of ablation importance.

        Returns
        -------
        DataFrame
        """
        n_points = min(100, len(x_test))
        sampled_index = x_test.sample(n_points, random_state=self._seed).index
        original_predictions = estimator_fn(x_test.loc[sampled_index])
        self._original_loss = self._loss(y_test.loc[sampled_index], original_predictions)

        x = x_test.loc[sampled_index]
        y = y_test.loc[sampled_index]

        results = pd.DataFrame.from_dict({
            seed: {
                permuted_column_name: self._loss(y.values, y_hat_permuted)
                for permuted_column_name, y_hat_permuted
                in self.__permutation_predictions(estimator_fn, x, seed).iteritems()
            }
            for seed in tqdm.tqdm(range(self._num_rounds))
        }, orient='index')
        results.index.name = 'seed'
        return results - self._original_loss, results


class PermutationImportanceSelectorKBest(PermutationImportanceSelector):
    def __init__(self, num_rounds=50, metric=mean_absolute_error, seed=42):
        super(PermutationImportanceSelectorKBest, self).__init__(k=1, num_rounds=num_rounds, loss_fn=metric, seed=seed)
        self._selector = SelectKBest(k=1)
        self.name = "pi4_1"
