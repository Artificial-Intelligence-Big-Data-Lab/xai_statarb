from datetime import date

import numpy as np

import utils


class WalkForward:
    def __init__(self, start_date, end_date, train_period_length, test_period_length=1, no_walks=None):
        self.start_date = start_date
        self.end_date = end_date
        self.train_period_length = train_period_length
        self.test_period_length = test_period_length
        self.no_walks = no_walks

    def __addYears(self, d, years):
        try:
            # Return same day of the current year
            return d.replace(year=d.year + years)
        except ValueError:
            # If not same day, it will return other, i.e.  February 29 to March 1 etc.
            return d + (date(d.year + years, 1, 1) - date(d.year, 1, 1))

    def get_walks(self):
        start_train = self.start_date
        idx = 0
        while (self.__addYears(start_train, self.train_period_length) < self.end_date and (
                idx < self.no_walks or self.no_walks is None)):
            idx = idx + 1
            yield idx, utils.Set(idx=idx, start=start_train, end=self.__addYears(start_train, self.train_period_length)), utils.Set(
                idx=idx, start=self.__addYears(start_train, self.train_period_length), end=np.min(
                    [self.__addYears(start_train, self.train_period_length + self.test_period_length), self.end_date]))
            start_train = self.__addYears(start_train, self.test_period_length)