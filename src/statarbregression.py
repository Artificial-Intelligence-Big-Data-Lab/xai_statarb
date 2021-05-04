import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from utils import *
from walkforward import Walk


class Environment:

    def __init__(self, tickers, args, base_folder='../LIME'):
        self.__base_folder = base_folder
        self.__test_folder = base_folder + '/test/'
        self.__file_names = [ticker + '.csv' if '.csv' not in ticker else ticker for ticker in tickers]
        self.__setup_folders(base_folder)
        self.__walk = None
        self.prediction_params = {
            'train': args.train_length,
            'val': args.validation_length,
            'test': args.test_length,
            'walks': args.no_walks,
        }

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
        return output_folder

    # def get_write_folder(self, data_type, ticker):
    #     if '.csv' not in ticker:
    #         ticker = ticker + '.csv'
    #
    #     write_path = self.data_folder + data_type + "_" + ticker.replace(" ", "_")
    #     print_info('Write path {0}'.format(write_path), file="stdout", flush=True)
    #     if os.path.exists(write_path):
    #         os.remove(write_path)
    #     return write_path

    # def get_folders(folder_path):
    #     folders = []
    #     stri = "./{0}/StatisticalArbitrage/*-*ARIMASINGLE2*_output".format(folder_path)
    #
    #     for folder in glob.glob(stri):
    #         print(folder)
    #         folders.append(folder)
    #     return folders

    def write_predictions_to_test(self, ticker, data: pd.DataFrame):
        data.to_csv("{0}{1}.csv".format(self.__test_folder, ticker))

    @staticmethod
    def _cleanup_folder(test_folder, files):
        for ticker in files:
            if os.path.exists(test_folder + ticker):
                os.remove(test_folder + ticker)

    def cleanup(self):
        self._cleanup_folder(self.test_folder, self.company_names)
        # self._cleanup_folder(self.data_folder, self.company_names)

    @property
    def company_names(self):
        return self.__file_names

    def __setup_folders(self, prediction_method):
        if not Path('./' + prediction_method + '/').exists():
            os.mkdir('./' + prediction_method + '/')

        test_folder = './' + prediction_method + '/test/'
        if not Path(test_folder).exists():
            os.mkdir(test_folder)

        self.__statistical_arbitrage_folder = './' + prediction_method + '/StatisticalArbitrage/'

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
    xspred = np.array(returns).cumsum()

    if len(xspred) == 0:
        return
    ipred = np.argmax(np.maximum.accumulate(xspred) - xspred)
    if ipred == 0:
        jpred = 0
    else:
        jpred = np.argmax(xspred[:ipred])

    mddpred = xspred[jpred] - xspred[ipred]
    return xspred, mddpred, ipred, jpred


class StatArbRegression:
    def __init__(self, test: pd.DataFrame, predicted_label='predicted', label='label', k=5, folder=None):
        self.__test = test.copy()
        if folder is not None and test.empty:
            self.__test = pd.read_csv(folder + '/totale.csv', ',', parse_dates=True)

        self.__k = k
        self.predicted_label = predicted_label
        self.label = label
        self.__columns = np.append(['ticker', self.label],
                                   [value for value in self.__test.columns if self.predicted_label in value])
        self.__methods = [value.replace(self.predicted_label, '').replace("_", "") for value in self.__test.columns if
                          self.predicted_label in value]

    def compute_long_short(self, label, ground_truth=False):
        active_range = set(range(0, self.__k, 1))

        long_pred = self.__test.sort_values([label], ascending=True).groupby(['Date']).nth(active_range)
        short_pred = self.__test.sort_values([label], ascending=False).groupby(['Date']).nth(active_range)

        long_bydate_pred = long_pred.groupby(['Date'])[self.label].agg([('value', 'sum'), ('no', 'count')])
        short_bydate_pred = short_pred.groupby(['Date'])[self.label].agg([('value', 'sum'), ('no', 'count')])

        long_bydate_pred = long_bydate_pred[long_bydate_pred['no'] >= self.__k]
        short_bydate_pred = short_bydate_pred[short_bydate_pred['no'] >= self.__k]

        valore_giornaliero_pred = (short_bydate_pred['value'] - long_bydate_pred['value']) / (
                long_bydate_pred['no'] + short_bydate_pred['no']) * 100

        return valore_giornaliero_pred, long_pred[['ticker', self.label, label]], short_pred[
            ['ticker', self.label, label]]

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

            short_bydate_pred[column + '_ticker'] = daily_short['ticker'].values
            short_bydate_pred[self.label + '_' + column] = daily_short[self.label].values
            short_bydate_pred[column] = daily_short[column].values

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
            outputfolder = kwargs.get('output_folder')

            self.__test[self.__columns].to_csv(outputfolder + '/totale.csv')

        return self.__valore_giornaliero_pred, self.__valore_giornaliero_exp

    def compute_metrics(self, **kwargs):
        print_info("compute metrics", file="stdout", flush=True)
        columns = [value for value in self.__test.columns if self.predicted_label in value]
        expected = self.__test[self.label].values

        metrics_dict = {}

        rmse_error = self.__test[columns].apply(lambda x: rmse(expected, x), axis=0)
        metrics_dict['MSE'] = dict(zip(columns, rmse_error))

        acc_error = self.__test[columns].apply(lambda x: mda(expected, x), axis=0)
        metrics_dict['MDA'] = dict(zip(columns, acc_error))

        if 'output_folder' in kwargs:
            output_folder = kwargs.get('output_folder')
            d = pd.DataFrame(data=metrics_dict)

            if output_folder is None:
                print(d)
            else:
                f = open(output_folder + '/metrics.csv', 'w+')
                f.write("Model metrics: \n" + str(d) + "\n")
                f.close()

    def plot_returns(self, output_folder, parameters):
        print_info(
            'shapes for predicted and expected daily returns {0} {1}'.format(str(self.__valore_giornaliero_pred.shape),
                                                                             str(self.__valore_giornaliero_exp.shape)),
            file="stdout", flush=True)
        if (
                self.__valore_giornaliero_pred is None or self.__valore_giornaliero_pred.empty or self.__valore_giornaliero_exp is None or self.__valore_giornaliero_exp.empty):
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


