from abc import ABC

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import lime
import lime.lime_tabular
from lime import submodular_pick


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


class RFFeatureImportanceSelector(FeatureSelectorBase):

    def fit_transform(self, estimator, X: pd.DataFrame, y: pd.DataFrame, X_test: pd.DataFrame,
                      y_test: pd.DataFrame):
        ####***************feature importance***********************
        print('*' * 20, 'feature importance', '*' * 20)
        all_columns = X.columns
        feat_imp_s = pd.DataFrame(
            {'features': all_columns, "feature_importance": estimator.feature_importances_}).sort_values(
            'feature_importance', ascending=False)
        column = feat_imp_s['features'].tail(self._k).values
        columns = set(all_columns) - set(column)
        self._importance = feat_imp_s[['features', 'feature_importance']].tail(self._k).values
        return columns


class PermutationImportanceSelector(FeatureSelectorBase):
    def __init__(self, k, num_rounds=50, metric=mean_squared_error, seed=0):
        super().__init__(k)
        self.__metric = metric
        self.__num_rounds = num_rounds
        self.__seed = seed

    def fit_transform(self, estimator, X: pd.DataFrame, y: pd.DataFrame, X_test: pd.DataFrame,
                      y_test: pd.DataFrame):
        ####***************permutation feature importance***********************
        print('*' * 20, 'permutation importance', '*' * 20)
        permutation_importance_s, _, _ = self.__compute_permutation_importance(X_test, y_test, estimator)
        min_row = permutation_importance_s['permutation_importance'].argsort()[:self._k]
        column = permutation_importance_s.iloc[min_row].features.values
        columns = set(X_test.columns) - set(column)

        self._importance = permutation_importance_s.iloc[min_row].reset_index().values

        return columns

    def __compute_permutation_importance(self, X_cr_test, y_cr_test, estimator):
        imp_values, all_trials = self.__feature_importance_permutation(
            predict_method=estimator.predict,
            X=X_cr_test.values,
            y=y_cr_test['label'].values,
            metric=self.__metric,
            seed=self.__seed)
        permutation_importance = pd.DataFrame(
            {'features': X_cr_test.columns.tolist(), "permutation_importance": imp_values}).sort_values(
            'permutation_importance', ascending=False)
        permutation_importance = permutation_importance.head(25)
        all_feat_imp_df = pd.DataFrame(data=np.transpose(all_trials), columns=X_cr_test.columns,
                                       index=range(0, self.__num_rounds))
        order_column = all_feat_imp_df.mean(axis=0).sort_values(ascending=False).index.tolist()
        return permutation_importance, all_feat_imp_df, order_column

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
                # importance = baseline - new_score
                mean_importance_values[col_idx] += importance
                all_importance_values[col_idx, round_idx] = importance
        mean_importance_values /= self.__num_rounds

        return mean_importance_values, all_importance_values


class LIMEPermutationImportance(FeatureSelectorBase):

    def __init__(self, k, num_exp_desired=126, random_state=123):
        super(LIMEPermutationImportance, self).__init__(k)
        self.__num_exp_desired = num_exp_desired
        self.__random_state = random_state

    def fit_transform(self, estimator, X, y, X_test, y_test):
        # ####***************LIME feature importance***********************
        print('*' * 20, 'LIME feature importance', '*' * 20)
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(X.values,
                                                                training_labels=y.values,
                                                                feature_names=X.columns.tolist(),
                                                                verbose=False, mode='regression'
                                                                , discretize_continuous=False
                                                                , random_state=self.__random_state)
        sp_obj_cr = submodular_pick.SubmodularPick(lime_explainer, X_test.values, estimator.predict,
                                                   num_features=len(X_test.columns),
                                                   num_exps_desired=self.__num_exp_desired)
        W_s = pd.DataFrame([dict(this.as_list(label=0)) for this in sp_obj_cr.explanations])
        rank_w_s = W_s[X_test.columns].abs().rank(1, ascending=False, method='first')
        rank_w_s_median, rank_w_s_mean = rank_w_s.median(), rank_w_s.mean()
        rank_w_s_median.name = 'median_rank'
        rank_w_s_mean.name = 'mean_rank'
        ranked_features = pd.concat([rank_w_s_median, rank_w_s_mean], axis=1).sort_values(
            by=['median_rank', 'mean_rank'],
            ascending=[False, False])
        min_row = ranked_features.index[:self._k].values
        columns = set(X_test.columns) - set(min_row)
        self._importance = ranked_features.head(self._k).reset_index().values
        return columns
