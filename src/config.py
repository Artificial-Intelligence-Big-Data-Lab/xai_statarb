from .feature_selection import RFFeatureImportanceSelector, LIMEFeatureImportanceSelector, PISelector, \
    WassersteinFeatureImportanceSelector

threshold_columns = {'walk', 'threshold_best', 'error_best', 'no_improvements_best', 'ratio_best',
                     'threshold_worst', 'error_worst', 'no_improvements_worst', 'ratio_worst',
                     'threshold_running', 'error_running', 'no_improvements_running', 'ratio_running'
                     }
thresholds_labels = ['best', 'worst', 'running']


def get_methods(args):
    return {
        'mdi': RFFeatureImportanceSelector(args.no_features),
        'sp': LIMEFeatureImportanceSelector(args.no_features),
        # 'pi': fs.PermutationImportanceSelector(features_no, seed=42),
        'pi': PISelector(seed=42, num_rounds=args.num_rounds),
        # 'pi_all': fs.PermutationImportanceSelector(seed=42),
        # 'pi3_all': fs.PISelectorUnormalized(seed=42),
        # 'pi_kl_all': fs.PIJensenShannonSelector(seed=42),
        'pi_wd': WassersteinFeatureImportanceSelector(seed=42),
        # "pi_mse": fs.PISelectorKBest(seed=42),
        # "pi_mae": fs.PermutationImportanceSelectorKBest(seed=42)
    }
