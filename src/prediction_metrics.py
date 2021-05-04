import pandas as pd

from .walkforward import Walk


class SelectedColumns:
    def __init__(self, save_path, test_run):
        self.__test_run = test_run
        self.__file = save_path + 'LOOC_metrics_cr_all_{0}.csv'.format(test_run)
        self.__feature_columns = None
        self.__all_columns = ['ticker', 'walk']
        self.__df = None

    @property
    def all_columns(self):
        return self.__all_columns

    @all_columns.setter
    def all_columns(self, value: []):
        if self.__feature_columns is None:
            self.__feature_columns = value
            self.__all_columns.extend(value)
        if self.__df is None:
            self.__df = pd.DataFrame(columns=self.__all_columns)

    def save(self):
        self.__df[self.__all_columns].to_csv(self.__file, index=False)
        return self.__df

    def set_chosen_features(self, ticker, walk: Walk, columns):
        if not columns:
            raise ValueError('At least one column must be selected')
        if not self.__feature_columns:
            raise ValueError('The feature names is not set')
        data_row = dict(ticker=ticker, walk=walk.train.idx)
        data_row.update(dict([(col, False) for col in self.__feature_columns]))
        data_row.update(dict([(col, True) for col in columns]))
        self.__df = self.__df.append(data_row, ignore_index=True)
