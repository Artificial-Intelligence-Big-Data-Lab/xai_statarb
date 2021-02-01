# Sebastian Raschka 2014-2020
# mlxtend Machine Learning Library Extensions
#
# Feature Importance Estimation Through Permutation
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import math
from collections import namedtuple
from sklearn.metrics import r2_score,max_error,explained_variance_score,mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_error, mean_squared_error

Set = namedtuple('Set', ['idx', 'start', 'end'])

class WalkForward:
  def __init__(self, start_date,end_date,train_period_length,test_period_length=1,no_walks=None):
    self.start_date=start_date
    self.end_date=end_date
    self.train_period_length=train_period_length
    self.test_period_length=test_period_length
    self.no_walks=no_walks    
                
  def __addYears(self, d, years):
    try:
        #Return same day of the current year        
        return d.replace(year = d.year + years)
    except ValueError:
        #If not same day, it will return other, i.e.  February 29 to March 1 etc.        
        return d + (date(d.year + years, 1, 1) - date(d.year, 1, 1))
      
  def get_walks(self):
     start_train = self.start_date
     idx=0
     while (self.__addYears(start_train,self.train_period_length)<self.end_date and (idx<self.no_walks or self.no_walks is None)):
       idx=idx+1
       yield idx, Set(idx=idx,start = start_train, end = self.__addYears(start_train,self.train_period_length) ), Set(idx=idx, start = self.__addYears(start_train,self.train_period_length), end = np.min([self.__addYears(start_train,self.train_period_length+self.test_period_length),self.end_date]))
       start_train = self.__addYears(start_train,self.test_period_length)

def feature_importance_permutation(X, y, predict_method, metric, num_rounds=1, seed=None):
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
    num_rounds : int (default=1)
        Number of rounds the feature columns are permuted to
        compute the permutation importance.
    seed : int or None (default=None)
        Random seed for permuting the feature columns.
    Returns
    ---------
    mean_importance_vals, all_importance_vals : NumPy arrays.
      The first array, mean_importance_vals has shape [n_features, ] and
      contains the importance values for all features.
      The shape of the second array is [n_features, num_rounds] and contains
      the feature importance for each repetition. If num_rounds=1,
      it contains the same values as the first array, mean_importance_vals.
    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/evaluate/feature_importance_permutation/
    """

    if not isinstance(num_rounds, int):
        raise ValueError('num_rounds must be an integer.')
    if num_rounds < 1:
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

    mean_importance_vals = np.zeros(X.shape[1])
    all_importance_vals = np.zeros((X.shape[1], num_rounds))

    baseline = score_func(y, predict_method(X))

    for round_idx in range(num_rounds):
        for col_idx in range(X.shape[1]):
            save_col = X[:, col_idx].copy()
            rng.shuffle(X[:, col_idx])
            new_score = score_func(y, predict_method(X))
            X[:, col_idx] = save_col
            if metric in list(['r2','accuracy']):
              importance = (baseline - new_score)/baseline
            else:
              importance = (new_score - baseline)/baseline
            # importance = baseline - new_score
            mean_importance_vals[col_idx] += importance
            all_importance_vals[col_idx, round_idx] = importance
    mean_importance_vals /= num_rounds

    return mean_importance_vals, all_importance_vals

def mda ( y_cr_test):
  """ Mean Directional Accuracy """
  
  x = np.sign( y_cr_test['label']-y_cr_test['label'].shift(1)) ==np.sign( y_cr_test['predicted']-y_cr_test['label'].shift(1))
  return np.count_nonzero(x.values.astype('int'))/len(x[~x.isnull()])

def hit_count(y_cr_test):
  x = (np.sign( y_cr_test['predicted'])==np.sign(y_cr_test['label'])).astype(int)
  return np.count_nonzero(x.values),np.count_nonzero(x.values)/len(x)

def get_prediction_performance_results(y_cr_test,show=True,prefix=''):
  results = pd.Series()
  metric_func={
      'MSE':mean_squared_error,
      'r2_score':r2_score,
      # 'explained_variance':explained_variance_score,
      'MAE':mean_absolute_error,
      # 'MAPE':mean_absolute_percentage_error
  }
  for metric,function in metric_func.items():
    column_name= '{0}_{1}'.format(prefix,metric) if prefix!='' else metric
    results[column_name]=function(y_cr_test['label'],y_cr_test['predicted'])
  results['{0}_MDA'.format(prefix) if prefix else 'MDA']=mda(y_cr_test)
  hc,acc=hit_count(y_cr_test)
  results['{0}_hit_count'.format(prefix) if prefix else 'hit_count']=hc
  results['{0}_accuracy'.format(prefix) if prefix else 'accuracy']=acc
  
  if (show):
    print(results)
  return results

def compute_permutation_importance(X_cr_test,y_cr_test,regressor, metric='r2',num_rounds = 50):
  imp_vals, all_trials = feature_importance_permutation(
      predict_method=regressor.predict, 
      X=X_cr_test.values,
      y=y_cr_test['label'].values,
      num_rounds=num_rounds, 
      metric=metric,
      seed=1)
  permutation_importance = pd.DataFrame({'features': X_cr_test.columns.tolist(), "permutation_importance": imp_vals}).sort_values('permutation_importance', ascending=False)
  permutation_importance = permutation_importance.head(25)
  all_feat_imp_df = pd.DataFrame(data=np.transpose(all_trials), columns=X_cr_test.columns, index = range(0,num_rounds))
  order_column = all_feat_imp_df.mean(axis=0).sort_values(ascending=False).index.tolist()
  return permutation_importance, all_feat_imp_df, order_column
