import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import print_info
from walkforward import Walk


class Environment:

    def __init__(self, tickers, args, sectors=None, base_folder='../LIME'):
        self.__base_folder = base_folder
        self.__test_folder = base_folder + '/{0}/test/'.format(args.test_no)

        self.__args = args

        if args.prediction_type == 'company' and (tickers is None or len(tickers) == 0):
            raise ValueError("Must provide at least a company to predict")
        if args.prediction_type == 'sector' and (sectors is None or len(sectors) == 0):
            raise ValueError("Must provide at least one sector to predict")
        if args.prediction_type == 'company':
            self.__file_names = [ticker + '.csv' if '.csv' not in ticker else ticker for ticker in tickers]
        else:
            self.__file_names = [sector + '.csv' if '.csv' not in sector else sector for sector in
                                 set(sectors) | set(tickers)]

        self.__setup_folders(self.__base_folder + '/{0}/'.format(args.test_no))
        self.__no_features = args.no_features.split(sep=',')
        self.__walk = None
        self.prediction_params = {
            'train': args.train_length,
            'val': args.validation_length,
            'test': args.test_length,
            'walks': args.no_walks,
        }
        f = open(self.__base_folder + '/{0}/input_parameters.txt'.format(self.__args.test_no), 'w+')
        f.write("Parameters: {0}\n".format(self.__args))
        f.close()

    @property
    def walk(self):
        return self.__walk

    @walk.setter
    def walk(self, value: Walk):
        self.__walk = value
        self.prediction_params.update(
            dict(train_start=value.train.start,
                 validation_start=value.validation.start,
                 test_start=value.test.start))

    @property
    def test_folder(self):
        return self.__test_folder

    @property
    def output_folder(self):
        output_folder = self.__statistical_arbitrage_folder + self.__walk.test.start.strftime('%Y-%m-%d')
        my_file = Path(output_folder)
        if not my_file.exists():
            os.mkdir(output_folder)
        for k in self.__no_features:
            my_file = Path(output_folder + '/{0}/'.format(k))
            if not my_file.exists():
                os.mkdir(output_folder + '/{0}/'.format(k))
        return output_folder

    @staticmethod
    def _cleanup_folder(test_folder, files):
        for ticker in files:
            if os.path.exists(test_folder + ticker):
                os.remove(test_folder + ticker)

    def cleanup(self):
        self._cleanup_folder(self.test_folder, self.__file_names)

    def __setup_folders(self, prediction_method):
        if not Path('./' + prediction_method + '/').exists():
            os.mkdir('./' + prediction_method + '/')

        test_folder = './' + prediction_method + '/test/'
        if not Path(test_folder).exists():
            os.mkdir(test_folder)

        self.__statistical_arbitrage_folder = './{0}/StatisticalArbitrage/'.format(prediction_method)

        if not Path(self.__statistical_arbitrage_folder).exists():
            os.mkdir(self.__statistical_arbitrage_folder)
        else:

            for the_file in os.listdir(self.__statistical_arbitrage_folder):
                file_path = os.path.join(self.__statistical_arbitrage_folder, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(e)


def compute_mdd(returns):
    xs_predicted = np.array(returns).cumsum()

    if len(xs_predicted) == 0:
        return
    ipred = np.argmax(np.maximum.accumulate(xs_predicted) - xs_predicted)
    if ipred == 0:
        jpred = 0
    else:
        jpred = np.argmax(xs_predicted[:ipred])

    mddpred = xs_predicted[jpred] - xs_predicted[ipred]
    return xs_predicted, mddpred, ipred, jpred


class StatArbRegression:
    def __init__(self, validation: pd.DataFrame, test: pd.DataFrame, predicted_label='predicted', label='label', k=5,
                 folder=None, prediction_type='company'):
        self.prediction_type = prediction_type
        self.__test = test.copy()
        self.__validation = validation.copy()
        if folder is not None and test.empty:
            self.__test = pd.read_csv(folder + '/totale.csv', ',', parse_dates=True)

        self.__k = k
        self.predicted_label = predicted_label
        self.label = label
        if prediction_type == 'company':
            self.__columns = np.append([self.label],
                                       [value for value in self.__test.columns if self.predicted_label in value])
        else:
            self.__columns = np.append(['sector', self.label],
                                       [value for value in self.__test.columns if self.predicted_label in value])
        self.__methods = [value.replace(self.predicted_label, '').replace("_", "") for value in self.__test.columns if
                          self.predicted_label in value]

    def compute_long_short(self, label, ground_truth=False):
        active_range = set(range(0, self.__k, 1))

        test_df = self.__test

        if 'ticker' in test_df.columns:
            test_df.drop(['ticker'], axis='columns', inplace=True)

        test_df = test_df.reset_index(level='ticker')

        long_pred = test_df.sort_values([label], ascending=False).groupby(['Date']).nth(active_range)
        short_pred = test_df.sort_values([label], ascending=True).groupby(['Date']).nth(active_range)

        long_bydate_pred = long_pred.groupby(['Date'])[self.label].agg([('value', 'sum'), ('no', 'count')])
        short_bydate_pred = short_pred.groupby(['Date'])[self.label].agg([('value', 'sum'), ('no', 'count')])

        long_bydate_pred = long_bydate_pred[long_bydate_pred['no'] >= self.__k]
        short_bydate_pred = short_bydate_pred[short_bydate_pred['no'] >= self.__k]

        valore_giornaliero_pred = (- short_bydate_pred['value'] + long_bydate_pred['value']) / (
                long_bydate_pred['no'] + short_bydate_pred['no']) * 100

        if 'sector' in long_pred.columns:
            return valore_giornaliero_pred, long_pred[['ticker', 'sector', self.label, label]], short_pred[
                ['ticker', 'sector', self.label, label]]
        else:
            return valore_giornaliero_pred, long_pred[['ticker', self.label, label]], short_pred[['ticker', self.label, label]]

    def __generate_signals(self, **kwargs):

        valore_giornaliero_exp, long_bydate_exp, short_bydate_exp = self.compute_long_short(self.label,
                                                                                            ground_truth=True)
        valore_giornaliero_pred, long_bydate_pred, short_bydate_pred = pd.DataFrame(
            index=valore_giornaliero_exp.index), pd.DataFrame(index=long_bydate_exp.index), pd.DataFrame(
            index=short_bydate_exp.index)

        columns = [value for value in self.__test.columns if self.predicted_label in value]

        for column in columns:
            daily_return, daily_long, daily_short = self.compute_long_short(column)
            valore_giornaliero_pred[column] = daily_return.values

            long_bydate_pred[self.label + '_' + column] = daily_long[self.label].values
            long_bydate_pred[column] = daily_long[column].values
            long_bydate_pred[column + '_ticker'] = daily_long['ticker'].values
            if 'sector' in daily_long.columns:
                long_bydate_pred[column + '_sector'] = daily_long['sector'].values

            short_bydate_pred[column + '_ticker'] = daily_short['ticker'].values
            short_bydate_pred[self.label + '_' + column] = daily_short[self.label].values
            short_bydate_pred[column] = daily_short[column].values
            if 'sector' in daily_short.columns:
                short_bydate_pred[column + '_sector'] = daily_short['sector'].values

        print_info('shapes for expected and predicted daily returns {0} {1}'.format(str(valore_giornaliero_pred.shape),
                                                                                    str(valore_giornaliero_exp.shape)),
                   file="stdout", flush=True)

        if 'output_folder' in kwargs:
            outputfolder = kwargs.get('output_folder')
            long_bydate_pred.to_csv(outputfolder + '/migliori_pred.csv')
            long_bydate_exp.to_csv(outputfolder + '/migliori_exp.csv')
            short_bydate_pred.to_csv(outputfolder + '/preggiori_pred.csv')
            short_bydate_exp.to_csv(outputfolder + '/peggiori_exp.csv')
            valore_giornaliero_exp.to_csv(outputfolder + '/valore_giornaliero_exp.csv', header=False)
            valore_giornaliero_pred.to_csv(outputfolder + '/valore_giornaliero_pred.csv')

        self.__valore_giornaliero_pred = valore_giornaliero_pred
        self.__valore_giornaliero_exp = valore_giornaliero_exp.loc[valore_giornaliero_pred.index]

        return valore_giornaliero_pred, valore_giornaliero_exp.loc[valore_giornaliero_pred.index]

    def generate_signals(self, **kwargs):
        r"""Generate signals and computes daily expected returns and daily actual returs of the strategy.
        Parameters
        ----------
        output_folder : string for dumping data to files.
        model_metrics : if passed will save the model metrics to file to the same output folder.
        Returns
        -------
        ret_exp: pandas data frame
            daily expected returns.
        ret_pred: pandas DataFrame
            daily realized returns.
        """
        print_info("generate signals", file="stdout", flush=True)
        self.__generate_signals(**kwargs)

        print_info(
            'shapes for predicted and expected daily returns {0} {1}'.format(str(self.__valore_giornaliero_pred.shape),
                                                                             str(self.__valore_giornaliero_exp.shape)),
            file="stdout", flush=True)

        if 'output_folder' in kwargs:
            output_folder = kwargs.get('output_folder')

            self.__test[self.__columns].to_csv(output_folder + '/totale.csv')
            self.__validation[self.__columns].to_csv(output_folder + '/validation_totale.csv')

        return self.__valore_giornaliero_pred, self.__valore_giornaliero_exp

    def plot_returns(self, output_folder, parameters):
        print_info(
            'shapes for predicted and expected daily returns {0} {1}'.format(str(self.__valore_giornaliero_pred.shape),
                                                                             str(self.__valore_giornaliero_exp.shape)),
            file="stdout", flush=True)
        if self.__valore_giornaliero_pred is None or self.__valore_giornaliero_pred.empty \
                or self.__valore_giornaliero_exp is None or self.__valore_giornaliero_exp.empty:
            self.__generate_signals(output_folder=output_folder)

        no_forecasts = len(self.__valore_giornaliero_exp)
        if no_forecasts == 0:
            return

        x_labstr = [el.strftime('%d-%m-%y') for el in pd.to_datetime(self.__valore_giornaliero_exp.index.values)]
        x_lab = np.arange(len(x_labstr)) + 1

        fig, fig_cumm = plt.figure(), plt.figure()
        ax_daily, ax_cumm = fig.add_subplot(111), fig_cumm.add_subplot(111)

        ax_daily.plot(x_lab, self.__valore_giornaliero_exp.values.tolist(), label='expected')

        ax_daily.hlines(0, 0, no_forecasts, colors='violet', linestyles='dashed', linewidth=0.3)
        for elx in x_lab:
            ax_daily.axvline(elx, color='yellow', linestyle='dashed', linewidth=0.3)

        f = open(output_folder + '/valori.txt', 'w+')

        for label in [value for value in self.__valore_giornaliero_pred.columns if self.predicted_label in value]:

            returns = self.__valore_giornaliero_pred[label].values.tolist()
            xspred, mddpred, ipred, jpred = compute_mdd(returns)

            f.write("Parameters: {0}\n".format(parameters))
            f.write("Method: {0}\n".format(label))

            f.write("Days: " + str(x_labstr) + "\n")
            f.write("return exp:" + str(sum(self.__valore_giornaliero_exp.values.tolist())) + "\n")

            f.write("return pred:" + str(sum(returns)) + "\n")
            f.write("mdd:" + str(mddpred) + "\n")
            f.write("Daily returns: " + str(returns) + "\n")
            f.write("Cumsum: " + str(xspred) + "\n")

            contprec = 0
            for elval in returns:
                print("ELVAL:", elval)
                if elval > 0:
                    contprec = contprec + 1
            f.write("Precision: {0}\n".format(str(float(contprec) / float(no_forecasts))))
            ax_daily.plot(x_lab, returns, label=label)

            ax_cumm.plot(xspred, label=label)
            ax_cumm.plot([ipred, jpred], [xspred[ipred], xspred[jpred]], 'o', color='Red', markersize=10)
        f.close()

        ax_daily.set(title='Daily returns')
        ax_daily.set_xticks(x_lab, x_labstr)
        ax_daily.legend()
        fig.savefig(output_folder + '/valore_percentuale.png')

        ax_cumm.set(title='return')
        ax_cumm.legend()
        fig_cumm.savefig(output_folder + '/curvareturn.png')
        plt.close()
