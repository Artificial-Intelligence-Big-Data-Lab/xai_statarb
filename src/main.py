import argparse
import datetime
import random
import time

from config import *
from feature_selection_threshold import Threshold, thresholds_labels
from get_model_input import *
from metrics import MetricsSaver, SelectedColumns
from models import get_fit_regressor
from statarbregression import *
from utils import get_prediction_performance_results, add_metrics_information, add_context_information, \
    init_prediction_df
from walkforward import WalkForward

BASE_PATH = '../LIME/'


def main(args):
    constituents = pd.read_csv(BASE_PATH + 'data/constituents_sp500.csv')
    # constituents = constituents.groupby(by=['Sector']).nth(set(range(0, 10, 1))).reset_index()
    # constituents = constituents[constituents['Sector'].isin(['Communication Services'])]
    # constituents = constituents[constituents['Ticker']=='ABBV']
    # tickers = set(tickers) | set(['ICE'])
    # tickers = ['UG.PA', 'CPG', 'FP.PA',
    #     '0001.HK', '0003.HK']

    random.seed(30)

    all_metrics_output_path = BASE_PATH + '{0}/LOOC_metrics_cr_all.csv'.format(args.test_no)

    env = Environment(tickers=constituents['Ticker'], sectors=constituents['Sector'].unique(), args=args)

    wf = WalkForward(datetime.datetime.strptime(args.start_date, '%Y-%m-%d'),
                     datetime.datetime.strptime(args.end_date, '%Y-%m-%d'),
                     train_period_length=args.train_length,
                     validation_period_length=args.validation_length,
                     test_period_length=args.test_length,
                     no_walks=args.no_walks)

    methods = get_methods(args)
    metrics_all = pd.DataFrame()
    company_feature_builder = CompanyFeatures(constituents=constituents, folder_output=env.test_folder,
                                              feature_type=args.data_type, prediction_type=args.prediction_type)
    chosen_columns = SelectedColumns(save_path=BASE_PATH + '{0}/'.format(args.test_no), removed_feature_no=args.no_features)
    metric_saver = MetricsSaver(labels=thresholds_labels)
    threshold_strategy = Threshold(thresholds_path=BASE_PATH + '{0}/'.format(args.test_no))

    for idx, walk in wf.get_walks():

        env.cleanup()
        env.walk = walk

        for ticker, constituents_batch in company_feature_builder.get_entities():

            print_info("{0}{1}{2}".format('*' * 20, ticker, '*' * 20))

            start_time = time.perf_counter()

            X_cr_train, y_cr_train, X_cr_validation, y_cr_validation, X_cr_test, y_cr_test = company_feature_builder.get_features(
                constituents_batch=constituents_batch, walk=walk)

            if len(X_cr_train) < 252 or len(X_cr_validation) == 0 or len(X_cr_test) == 0:
                continue

            chosen_columns.all_columns = X_cr_train.columns

            context = dict(walk=idx, ticker=ticker, method='baseline', start=walk.train.start, end=walk.train.end,
                           all_columns=X_cr_validation.columns)
            baseline, b_y_validation, b_y_test, score = get_fit_regressor(X_cr_train, y_cr_train,
                                                                          x_validation=X_cr_validation,
                                                                          y_validation=y_cr_validation,
                                                                          x_test=X_cr_test, y_cr_test=y_cr_test,
                                                                          data_type=args.data_type,
                                                                          model_type=args.model_type,
                                                                          prediction_type=args.prediction_type,
                                                                          get_cross_validation_results=False)

            metric_single_baseline = get_prediction_performance_results(b_y_validation, False)
            metrics_baseline = add_metrics_information(metric_single_baseline, context, score)
            end_time = time.perf_counter()
            print_info('{0} training took {1} s'.format(ticker, end_time - start_time))
            metric_single_baseline, _ = add_context_information(metric_single_baseline, context, score)

            for method, transformer in {args.method: methods[args.method]}.items():
                for col_idx, importance, columns, selection_error in transformer.fit_transform(baseline, X_cr_train,
                                                                                               y_cr_train,
                                                                                               X_cr_validation,
                                                                                               y_cr_validation):
                    looc_fi_regressor, looc_y_validation, looc_y_test, score_looc = get_fit_regressor(X_cr_train,
                                                                                                      y_cr_train,
                                                                                                      x_validation=X_cr_validation,
                                                                                                      y_validation=y_cr_validation,
                                                                                                      x_test=X_cr_test,
                                                                                                      y_cr_test=y_cr_test,
                                                                                                      data_type=args.data_type,
                                                                                                      model_type=args.model_type,
                                                                                                      prediction_type=args.prediction_type,
                                                                                                      columns=columns,
                                                                                                      get_cross_validation_results=False)
                    metrics_fi_looc = get_prediction_performance_results(looc_y_validation, False)

                    context.update(dict(method=method, selection_error=selection_error, index=col_idx))
                    merged_series = metrics_baseline.copy()
                    merged_series = add_metrics_information(metrics_fi_looc, context, score_looc,
                                                            importance_series=importance, copy_to=merged_series)
                    _, _ = add_context_information(metrics_fi_looc, context, score_looc,
                                                   importance_series=importance,
                                                   baseline_loss=transformer.baseline_loss)

                    metrics_all = metrics_all.append(pd.DataFrame(merged_series).T, ignore_index=True)
                    metrics_all.to_csv(all_metrics_output_path, index=False)

            end_time = time.perf_counter()
            print_info('{0} took {1} s'.format(ticker, end_time - start_time))

        dfs = threshold_strategy.get_thresholds(metrics_all, walk.train.idx)

        print_info('*' * 10 + 'START forecasting using optimal threshold' + '*' * 10)
        total_df = pd.DataFrame()
        validation_total_df = pd.DataFrame()

        for ticker, constituents_batch in company_feature_builder.get_entities():

            X_cr_train, y_cr_train, X_cr_validation, y_cr_validation, X_cr_test, y_cr_test = company_feature_builder.get_features(
                constituents_batch=constituents_batch, walk=walk)

            if len(X_cr_train) < 252 or len(X_cr_validation) == 0 or len(X_cr_test) == 0:
                continue

            baseline, b_y_validation, b_y_test, score = get_fit_regressor(X_cr_train, y_cr_train,
                                                                          x_validation=X_cr_validation,
                                                                          y_validation=y_cr_validation,
                                                                          x_test=X_cr_test, y_cr_test=y_cr_test,
                                                                          data_type=args.data_type,
                                                                          model_type=args.model_type,
                                                                          prediction_type=args.prediction_type,
                                                                          get_cross_validation_results=False,
                                                                          suffix='_baseline')
            predictions_df = init_prediction_df(ticker, X_cr_test, y_cr_test, b_y_test, args.prediction_type)
            validation_predictions_df = init_prediction_df(ticker, X_cr_validation, y_cr_validation, b_y_validation, args.prediction_type)

            for th_label in thresholds_labels:
                columns = chosen_columns.get_columns(dfs[th_label], ticker, method=th_label)
                chosen_columns.set_chosen_features(ticker=ticker, walk_idx=walk.train.idx, method=th_label, columns=columns)

                looc_fi_regressor, looc_y_validation, looc_y_cr_test, score_looc = get_fit_regressor(X_cr_train,
                                                                                                     y_cr_train,
                                                                                                     x_validation=X_cr_validation,
                                                                                                     y_validation=y_cr_validation,
                                                                                                     x_test=X_cr_test,
                                                                                                     y_cr_test=y_cr_test,
                                                                                                     data_type=args.data_type,
                                                                                                     model_type=args.model_type,
                                                                                                     prediction_type=args.prediction_type,
                                                                                                     columns=columns,
                                                                                                     get_cross_validation_results=False,
                                                                                                     suffix="_" + th_label)

                predictions_df = predictions_df.join(looc_y_cr_test)
                validation_predictions_df = validation_predictions_df.join(looc_y_validation)

            if not predictions_df.empty and not validation_predictions_df.empty:
                total_df = pd.concat([total_df, predictions_df], axis=0)
                validation_total_df = pd.concat([validation_total_df, validation_predictions_df], axis=0)
                metric_saver.set_metrics(ticker, idx, validation_predictions_df, predictions_df)

        chosen_columns.save()
        metric_saver.save(env.output_folder)

        print_info('*' * 10 + 'END forecasting using optimal threshold' + '*' * 10)
        strategy1 = StatArbRegression(validation=validation_total_df, test=total_df, predicted_label='predicted', k=args.k,
                                      prediction_type=args.prediction_type)
        strategy1.generate_signals(output_folder=env.output_folder)
        strategy1.plot_returns(output_folder=env.output_folder, parameters=env.prediction_params)

    print('*' * 20, 'DONE', '*' * 20)
    metrics_all.to_csv(all_metrics_output_path, index=False)
    env.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--method',
                        choices=['mdi', 'sp', 'pi', 'pi_wd'],
                        default='pi',
                        type=str)

    parser.add_argument('--model_type',
                        choices=['rf', 'svr', 'lgb'],
                        default='rf',
                        type=str)

    parser.add_argument('--prediction_type',
                        choices=['company', 'sector'],
                        default='sector',
                        type=str)

    parser.add_argument('--data_type',
                        choices=['cr', 'cr_ti', 'ti'],
                        default='cr_ti',
                        type=str)

    parser.add_argument('--start_date',
                        help='Start date "%Y-%m-%d" format',
                        default='2003-01-01',
                        type=str)
    parser.add_argument('--end_date',
                        help='End date "%Y-%m-%d" format',
                        default='2018-01-01',
                        type=str)
    parser.add_argument('--no_walks',
                        help='Number of walks',
                        default='2',
                        type=int)
    parser.add_argument('--no_features',
                        help='Number of features to remove',
                        default=1,
                        type=int)
    parser.add_argument('--train_length',
                        help='Number of training data expressed in years (Y) or months (M). Default value 4Y ',
                        default='3Y',
                        type=str)
    parser.add_argument('--validation_length',
                        help='Number of validation data expressed in years(Y) or months (M). Default value 1Y',
                        default='1Y',
                        type=str)
    parser.add_argument('--test_length',
                        help='Number of test data expressed in years(Y) or months (M). Default value 1Y',
                        default='1Y',
                        type=str)
    parser.add_argument('--k',
                        help='StatArb number of companies',
                        default=5,
                        type=int)
    parser.add_argument('--num_rounds',
                        help='Number of permutation rounds',
                        default=51,
                        type=int)
    parser.add_argument('--test_no',
                        help='Test number to identify the experiments',
                        default=48,
                        type=int)
    args_in = parser.parse_args()

    main(args_in)
