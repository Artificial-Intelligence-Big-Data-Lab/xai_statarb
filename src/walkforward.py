from collections import namedtuple

import numpy as np
from dateutil.relativedelta import relativedelta

Set = namedtuple('Set', ['idx', 'start', 'end'])

Walk = namedtuple('Walk', ['train', 'validation', 'test'])


class WalkForward:
    def __init__(self, start_date, end_date=None, train_period_length='4Y', validation_period_length='1Y',
                 test_period_length='1Y', no_walks=None):
        """

        Parameters
        ----------
        end_date : datetime
        start_date : datetime
        """
        if end_date is None and no_walks is None:
            raise ValueError('Either end date or number of walks must be specified')

        self.__train_months = 4 * 12
        if 'M' in train_period_length.upper():
            self.__train_months = int(train_period_length.upper().replace('M', ""))
        elif 'Y' in train_period_length.upper():
            self.__train_months = int(train_period_length.upper().replace('Y', '')) * 12

        self.__validation_months = 1 * 12
        if 'M' in validation_period_length.upper():
            self.__validation_months = int(validation_period_length.upper().replace('M', ""))
        elif 'Y' in train_period_length.upper():
            self.__validation_months = int(validation_period_length.upper().replace('Y', '')) * 12

        self.__test_months = 1 * 12
        if 'M' in test_period_length.upper():
            self.__test_months = int(test_period_length.upper().replace('M', ""))
        elif 'Y' in train_period_length.upper():
            self.__test_months = int(test_period_length.upper().replace('Y', '')) * 12

        if end_date is None:
            end_date = start_date + relativedelta(
                months=(self.__validation_months + self.__train_months) * no_walks + self.__test_months)

        self.start_date = start_date
        self.end_date = end_date
        self.train_period_length = train_period_length
        self.validation_period_length = validation_period_length
        self.test_period_length = test_period_length
        self.no_walks = no_walks

    def get_walks(self):
        start_train = self.start_date
        idx = 0
        start_validation = start_train + relativedelta(months=+self.__train_months)

        start_test = start_validation + relativedelta(months=+self.__validation_months)

        end_test = start_test + relativedelta(months=+self.__test_months) + relativedelta(days=-1)

        while (end_test < self.end_date) and (idx < self.no_walks or self.no_walks is None):
            idx = idx + 1
            walk = Walk(train=Set(idx=idx, start=start_train, end=start_validation + relativedelta(days=-1)), \
                        validation=Set(idx=idx, start=start_validation, end=start_test + relativedelta(days=-1)), \
                        test=Set(idx=idx, start=start_test, end=np.min([end_test, self.end_date])))
            print('*' * 20, idx, '*' * 20)
            print(walk.train.start, walk.train.end)
            print(walk.validation.start, walk.validation.end)
            print(walk.test.start, walk.test.end)
            print('*' * 20)
            yield idx, walk

            start_train = start_train + relativedelta(months=+self.__test_months)
            start_validation = start_train + relativedelta(months=+self.__train_months)
            start_test = start_validation + relativedelta(months=+self.__validation_months)
            end_test = start_test + relativedelta(months=+self.__test_months) + relativedelta(days=-1)
