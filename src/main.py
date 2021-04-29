import argparse
import datetime
import random

import feature_selection as fs
from feature_selection_threshold import *
from get_model_input import *
from models import get_fit_regressor
from statarbregression import *
from walkforward import WalkForward

DATA_PATH = '../LIME/data/'


def main(args):
    constituents = pd.read_csv(DATA_PATH + 'constituents.csv')
    tickers = constituents['Ticker']
    tickers = ['FP.PA', '0001.HK', '0003.HK']

    random.seed(30)

    metrics_output_path = DATA_PATH + 'LOOC_metrics_cr_{0}.csv'.format(args.test_no)
    all_metrics_output_path = DATA_PATH + 'LOOC_metrics_cr_all_{0}.csv'.format(args.test_no)
    removed_columns_path = DATA_PATH + 'LOOC_missing_columns_cr_{0}.csv'.format(args.test_no)
    thresholds_path = DATA_PATH + 'LOOC_thresholds_{0}.csv'.format(args.test_no)

    thresholds = pd.DataFrame(columns={'walk', 'threshold_best', 'error_best', 'no_improvements_best', 'ratio_best'
        , 'threshold_worst', 'error_worst', 'no_improvements_worst', 'ratio_worst'
        , 'threshold_running', 'error_running', 'no_improvements_running', 'ratio_running'
                                       })

    env = Environment(tickers=tickers, args=args)

    wf = WalkForward(datetime.datetime.strptime(args.start_date, '%Y-%m-%d'),
                     datetime.datetime.strptime(args.end_date, '%Y-%m-%d'),
                     train_period_length=args.train_length,
                     validation_period_length=args.validation_length,
                     test_period_length=args.test_length,
                     no_walks=args.no_walks)

    methods = {
        'mdi': fs.RFFeatureImportanceSelector(args.no_features),
        'sp': fs.LIMEFeatureImportanceSelector(args.no_features),
        # 'pi': fs.PermutationImportanceSelector(features_no, seed=42),
        'pi': fs.PISelector(seed=42, num_rounds=args.num_rounds),
        # 'pi_all': fs.PermutationImportanceSelector(seed=42),
        # 'pi3_all': fs.PISelectorUnormalized(seed=42),
        # 'pi_kl_all': fs.PIJensenShannonSelector(seed=42),
        'pi_wd': fs.WassersteinFeatureImportanceSelector(seed=42),
        # "pi_mse": fs.PISelectorKBest(seed=42),
        # "pi_mae": fs.PermutationImportanceSelectorKBest(seed=42)
    }
    metrics = pd.read_csv(metrics_output_path) if os.path.exists(metrics_output_path) else pd.DataFrame()
    metrics_all = pd.DataFrame()

    missing_columns = pd.read_csv(removed_columns_path) if os.path.exists(removed_columns_path) else pd.DataFrame()

    for idx, walk in wf.get_walks():
        print('*' * 20, idx, '*' * 20)
        print(walk.train.start, walk.train.end)
        print(walk.validation.start, walk.validation.end)
        print(walk.test.start, walk.test.end)
        print('*' * 20)
        env.cleanup()
        env.walk = walk

        start_test = walk.test.start
        validation_start = walk.validation.start

        for ticker in tickers:

            print('*' * 20, ticker, '*' * 20)

            if check_if_processed(metrics, ticker, idx):
                print("ticker {0} ALREADY PROCESSED".format(ticker))
                continue

            start_time = time.perf_counter()
            x_df, y_df = get_data_for_ticker(ticker, walk.train.start, walk.test.end)

            if x_df.empty or y_df.empty:
                continue

            X_cr_train, y_cr_train = x_df.loc[:validation_start], y_df.loc[:validation_start]
            X_cr_validation, y_cr_validation = x_df.loc[validation_start:start_test], y_df.loc[
                                                                                      validation_start:start_test]

            if len(X_cr_train) == 0 or len(y_cr_validation) == 0:
                continue
            print('{0} train {1} {2}'.format(ticker, X_cr_train.index.min(), X_cr_train.index.max()))
            print('{0} test {1} {2}'.format(ticker, X_cr_validation.index.min(), X_cr_validation.index.max()))

            context = dict(walk=idx, ticker=ticker, method='baseline', start=walk.train.start, end=walk.train.end,
                           all_columns=X_cr_validation.columns)
            baseline, b_y_cr_test, score = get_fit_regressor(X_cr_train, y_cr_train, X_cr_validation, y_cr_validation)

            metric_single_baseline = get_prediction_performance_results(b_y_cr_test, False)
            metrics_baseline = add_metrics_information(metric_single_baseline, context, score)

            metric_single_baseline, _ = add_context_information(metric_single_baseline, context, score)

            metrics = metrics.append(metric_single_baseline, ignore_index=True)
            metrics.to_csv(metrics_output_path, index=False)

            for method, transformer in {args.data_type: methods[args.data_type]}.items():
                for col_idx, importance, columns, selection_error in transformer.fit_transform(baseline, X_cr_train,
                                                                                               y_cr_train,
                                                                                               X_cr_validation,
                                                                                               y_cr_validation):

                    looc_fi_regressor, looc_y_cr_test, score_looc = get_fit_regressor(X_cr_train, y_cr_train,
                                                                                      X_cr_validation,
                                                                                      y_cr_validation,
                                                                                      columns=columns)
                    metrics_fi_looc = get_prediction_performance_results(looc_y_cr_test, False)

                    context.update(dict(method=method, selection_error=selection_error, index=col_idx))
                    merged_series = metrics_baseline.copy()
                    merged_series = add_metrics_information(metrics_fi_looc, context, score_looc,
                                                            importance_series=importance, copy_to=merged_series)
                    metrics_fi_looc, missing_col_dict = add_context_information(metrics_fi_looc, context, score_looc
                                                                                , importance_series=importance,
                                                                                baseline_loss=transformer.baseline_loss)

                    if missing_col_dict is not None and not missing_col_dict.empty:
                        missing_columns = missing_columns.append(missing_col_dict, ignore_index=True)
                        missing_columns.to_csv(removed_columns_path, index=False)

                    metrics = metrics.append(metrics_fi_looc, ignore_index=True)
                    metrics.to_csv(metrics_output_path, index=False)

                    metrics_all = metrics_all.append(pd.DataFrame(merged_series).T, ignore_index=True)
                    metrics_all.to_csv(all_metrics_output_path, index=False)

            end_time = time.perf_counter()
            print('{0} took {1} s'.format(ticker, end_time - start_time))

        fix_th_best = get_errors_df_by_walk_5(metrics_all, np.arange(0.0, -0.03, -0.0001), idx, worst=False)
        fix_th_worst = get_errors_df_by_walk_5(metrics_all, np.arange(0.0, -0.03, -0.0001), idx, worst=True)
        running_th = get_errors_df_by_walk_3(metrics_all, np.arange(0.0, -0.03, -0.0001), idx, worst=False)
        threshold_row = get_optimal_threshold(fix_th_worst, fix_th_best, running_th, idx)
        thresholds = thresholds.append(threshold_row, ignore_index=True)
        thresholds.to_csv(thresholds_path, index=False)

        total_df = pd.DataFrame()
        for ticker in tickers:
            x_df, y_df = get_data_for_ticker(ticker, walk.train.start, walk.test.end)

            if x_df.empty or y_df.empty:
                continue

            X_cr_train, y_cr_train = x_df.loc[:validation_start], y_df.loc[:validation_start]
            X_cr_validation, y_cr_validation = x_df.loc[validation_start:start_test], y_df.loc[
                                                                                      validation_start:start_test]
            X_cr_test, y_cr_test = x_df.loc[start_test:], y_df.loc[start_test:]

            if len(X_cr_train) == 0 or len(y_cr_validation) == 0:
                continue
            baseline, b_y_cr_test, score = get_fit_regressor(X_cr_train, y_cr_train, X_cr_test, y_cr_test)

            looc_fi_regressor, looc_y_cr_test, score_looc = get_fit_regressor(X_cr_train, y_cr_train,
                                                                              X_cr_test, y_cr_test,
                                                                              columns=columns)
            df = X_cr_test.copy()
            df = pd.concat([df, y_cr_test], axis=1)
            df = df.join(b_y_cr_test, rsuffix="_baseline")
            df = df.join(looc_y_cr_test, rsuffix="_best")
            if not df.empty:
                total_df = pd.concat([total_df, df], axis=0)

        strategy1 = StatArbRegression(total_df, 'predicted')
        strategy1.compute_metrics(output_folder=env.output_folder)
        strategy1.generate_signals(output_folder=env.output_folder)
        strategy1.plot_returns(output_folder=env.output_folder, parameters=env.prediction_params)

    print('*' * 20, 'DONE', '*' * 20)
    metrics.to_csv(metrics_output_path, index=False)
    metrics_all.to_csv(all_metrics_output_path, index=False)
    thresholds.to_csv(thresholds_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_type',
        choices=['mdi', 'sp', 'pi', 'pi_wd'],
        default='pi',
        type=str)

    parser.add_argument(
        '--start_date',
        help='Start date "%Y-%m-%d" format',
        default='2007-01-01',
        type=str)
    parser.add_argument(
        '--end_date',
        help='End date "%Y-%m-%d" format',
        default='2018-01-01',
        type=str)
    parser.add_argument(
        '--no_walks',
        help='Number of walks',
        default='7',
        type=int)
    parser.add_argument(
        '--no_features',
        help='Number of features to remove',
        default=1,
        type=int)
    parser.add_argument(
        '--train_length',
        help='Number of training data expressed in years (Y) or months (M). Default value 4Y ',
        default='4Y',
        type=str)
    parser.add_argument(
        '--validation_length',
        help='Number of validation data expressed in years(Y) or months (M). Default value 1Y',
        default='1Y',
        type=str)
    parser.add_argument(
        '--test_length',
        help='Number of test data expressed in years(Y) or months (M). Default value 1Y',
        default='1Y',
        type=str)
    parser.add_argument(
        '--num_rounds',
        help='Number of permutation rounds',
        default=50,
        type=int)
    parser.add_argument(
        '--test_no',
        help='Test number to identify the experiments',
        default=11,
        type=int)
    args_in = parser.parse_args()

    main(args_in)
