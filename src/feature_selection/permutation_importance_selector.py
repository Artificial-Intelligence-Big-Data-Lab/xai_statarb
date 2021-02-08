import numpy as np
import pandas as pd
import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .feature_selector_base import *


class PISelector(FeatureSelectorBase):
    def __init__(self, k=0, num_rounds=50, metric=mean_squared_error, seed=0):
        super().__init__(k)
        self.__metric = metric
        self.__num_rounds = num_rounds
        self.__seed = seed
        self.__selector = AllAboveZeroSelector() if k == 0 else SelectKBest(k)

    def fit_transform(self, estimator, X: pd.DataFrame, y: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame):
        print('*' * 20, 'permutation importance', '*' * 20)
        permutation_importance_s = self.__compute_permutation_importance(X_test, y_test, estimator)
        self.__selector.importance_ = permutation_importance_s
        min_row, columns = self.__selector.select(X_test.columns)
        self._importance = min_row.reset_index()
        return columns

    def __compute_permutation_importance(self, X_test, y_cr_test, estimator):
        imp_values, all_trials, all_trial_errors = self.__feature_importance_permutation(
            predict_method=estimator.predict,
            X=X_test.values,
            y=y_cr_test['label'].values,
            metric=self.__metric,
            seed=self.__seed)
        all_feat_imp_df = pd.DataFrame(data=np.transpose(all_trials), columns=X_test.columns,
                                       index=range(0, self.__num_rounds))
        all_trial_errors_df = pd.DataFrame(data=np.transpose(all_trial_errors), columns=X_test.columns,
                                           index=range(0, self.__num_rounds))
        imp_ci = 1.96 * all_feat_imp_df.std(ddof=0) / np.sqrt(self.__num_rounds)
        permutation_importance = pd.DataFrame(
            dict(
                feature_importance=all_feat_imp_df.mean(),
                ci_fixed=imp_ci,
                errors=all_trial_errors_df.mean(),
                std_errors=all_trial_errors_df.std()
            ),
        )

        return permutation_importance

    def __feature_importance_permutation(self, X, y, predict_method, metric, seed):
        """Feature importance imputation via permutation importance
        Parameters
        ----------
        X : NumPy array, shape = [n_samples, n_features]
            Dataset, where n_samples is the number of samples and
            n_features is the number of features.
        y : NumPy array, shape = [n_samples]
            Target values.
        predict_method : prediction function
            A callable function that predicts the target values
            from X.
        metric : str, callable
            The metric for evaluating the feature importance through
            permutation. By default, the strings 'accuracy' is
            recommended for classifiers and the string 'r2' is
            recommended for regressors. Optionally, a custom
            scoring function (e.g., `metric=scoring_func`) that
            accepts two arguments, y_true and y_pred, which have
            similar shape to the `y` array.
        seed : int or None (default=None)
            Random seed for permuting the feature columns.
        Returns
        ---------
        mean_importance_values, all_importance_values : NumPy arrays.
          The first array, mean_importance_values has shape [n_features, ] and
          contains the importance values for all features.
          The shape of the second array is [n_features, num_rounds] and contains
          the feature importance for each repetition. If num_rounds=1,
          it contains the same values as the first array, mean_importance_vals.
        Examples
        -----------
        For usage examples, please see
        http://rasbt.github.io/mlxtend/user_guide/evaluate/feature_importance_permutation/
        """

        if not isinstance(self.__num_rounds, int):
            raise ValueError('num_rounds must be an integer.')
        if self.__num_rounds < 1:
            raise ValueError('num_rounds must be greater than 1.')

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

        rng = np.random.RandomState(seed)

        mean_importance_values = np.zeros(X.shape[1])
        all_importance_values = np.zeros((X.shape[1], self.__num_rounds))
        all_error_values = np.zeros((X.shape[1], self.__num_rounds))

        baseline = score_func(y, predict_method(X))

        for round_idx in range(self.__num_rounds):
            for col_idx in range(X.shape[1]):
                save_col = X[:, col_idx].copy()
                rng.shuffle(X[:, col_idx])
                new_score = score_func(y, predict_method(X))
                X[:, col_idx] = save_col
                if metric in list(['r2', 'accuracy']):
                    importance = (baseline - new_score) / baseline
                else:
                    importance = (new_score - baseline) / baseline
                mean_importance_values[col_idx] += importance
                all_importance_values[col_idx, round_idx] = importance
                all_error_values[col_idx, round_idx] = new_score

        mean_importance_values /= self.__num_rounds

        return mean_importance_values, all_importance_values, all_error_values


class PermutationImportanceSelector(FeatureSelectorBase):
    def __init__(self, k=0, num_rounds=50, loss_fn=mean_absolute_error, seed=0):
        super().__init__(k)
        self.__loss_fn = loss_fn
        self.__num_rounds = num_rounds
        self.__seed = seed
        self.__selector = AllAboveZeroSelector() if k == 0 else SelectKBest(k)

    def fit_transform(self, estimator, X: pd.DataFrame, y: pd.DataFrame, X_test: pd.DataFrame,
                      y_test: pd.DataFrame):
        print('*' * 20, 'permutation importance', '*' * 20)

        n_points = min(100, len(X_test))
        sampled_index = X_test.sample(n_points, random_state=self.__seed).index
        original_predictions = estimator.predict(X_test.loc[sampled_index])
        self.__original_loss = self.__loss_fn(y_test.loc[sampled_index], original_predictions)
        res, error_res = self.__ablation_importance(
            model_fn=estimator.predict,
            x=X_test.loc[sampled_index],
            y=y_test.loc[sampled_index],
            n_repetitions=self.__num_rounds,
        )
        mean_imp = res.mean()

        # 95% CI based upon z-distribution approximation
        # of t-distribution using MLE estimate of variance
        imp_ci = 1.96 * res.std(ddof=0) / np.sqrt(self.__num_rounds)

        imp = pd.DataFrame(
            dict(
                feature_importance=mean_imp,
                ci_fixed=imp_ci,
                errors=error_res.mean(),
                std_errors=error_res.std()
            ),
        )
        self.__selector.importance_ = imp
        min_row, columns = self.__selector.select(X_test.columns)
        self._importance = min_row.reset_index()
        return columns

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

    def __ablation_importance(self, model_fn, x, y, n_repetitions=50):
        """Reference implementation of ablation importance.

        Returns
        -------
        DataFrame
        """

        results = pd.DataFrame.from_dict({
            seed: {
                permuted_column_name: self.__loss_fn(y, y_hat_permuted)
                for permuted_column_name, y_hat_permuted
                in self.__permutation_predictions(model_fn, x, seed).iteritems()
            }
            for seed in tqdm.tqdm(range(n_repetitions))
        }, orient='index')
        results.index.name = 'seed'
        return results-self.__original_loss, results
