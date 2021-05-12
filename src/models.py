import lightgbm as lgb
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

company_model_parameters = dict(rf=dict(cr=dict(n_estimators=500, min_samples_leaf=5, max_features=1, oob_score=True),
                                        ti=dict(n_estimators=150, max_depth=12, max_features=1, oob_score=True)),
                                svr=dict(cr=dict(max_iter=1e6, C=0.005, tol=1e-3, epsilon=0.002),
                                         ti=dict(max_iter=1e6, C=0.5, tol=1e-4, gamma='scale', epsilon=1e-3),
                                         # ti=dict(max_iter=1e6, C=0.2, tol=1e-3, gamma='scale', epsilon=1e-4)))
                                         ),
                                lgb=dict(ti=dict(
                                    # n_estimators=1000, n_jobs=-1, max_depth=10
                                    # , learning_rate=5e-3
                                    # , colsample_bytree=.2
                                    # , subsample=0.3
                                    # # ,boosting_type='goss'
                                    # n_estimators=500, n_jobs=-1
                                    # , learning_rate=5e-2
                                    # , colsample_bytree=.3
                                    # , subsample=0.8
                                    # , bagging_fraction=0.3
                                    # , bagging_freq=21
                                    # , num_leaves=21
                                    # n_estimators=1000
                                    # # ,max_depth=12
                                    # , learning_rate=5e-2
                                    # , colsample_bytree=.5
                                    # , subsample=0.1
                                    # , bagging_fraction=0.2
                                    # , bagging_freq=5
                                    # , num_leaves=21
                                    num_leaves=21,
                                    n_estimators=100,
                                    learning_rate=5e-2,
                                    max_depth=8,
                                    colsample_bytree=.2,
                                    subsample=.8,
                                    reg_alpha=1e-3,
                                ),
                                    cr=dict(n_estimators=500, learning_rate=5e-2
                                            , bagging_fraction=0.1, bagging_freq=5)
                                )
                                )

sector_model_parameters = dict(rf=dict(cr=dict(n_estimators=500, min_samples_leaf=5, max_features=1, oob_score=True),
                                       ti=dict(n_estimators=150, max_depth=12, max_features=1, oob_score=True)),
                               svr=dict(cr=dict(max_iter=1e6, C=0.005, tol=1e-3, epsilon=0.002),
                                        ti=dict(max_iter=1e6, C=0.5, tol=1e-4, gamma='scale', epsilon=1e-3),
                                        # ti=dict(max_iter=1e6, C=0.2, tol=1e-3, gamma='scale', epsilon=1e-4)))
                                        ),
                               lgb=dict(ti=dict(
                                   # n_estimators=1000, n_jobs=-1, max_depth=10
                                   # , learning_rate=5e-3
                                   # , colsample_bytree=.2
                                   # , subsample=0.3
                                   # # ,boosting_type='goss'
                                   # n_estimators=500, n_jobs=-1
                                   # , learning_rate=5e-2
                                   # , colsample_bytree=.3
                                   # , subsample=0.8
                                   # , bagging_fraction=0.3
                                   # , bagging_freq=21
                                   # , num_leaves=21
                                   # n_estimators=1000
                                   # # ,max_depth=12
                                   # , learning_rate=5e-2
                                   # , colsample_bytree=.5
                                   # , subsample=0.1
                                   # , bagging_fraction=0.2
                                   # , bagging_freq=5
                                   # , num_leaves=21
                                   num_leaves=21,
                                   n_estimators=100,
                                   learning_rate=5e-2,
                                   max_depth=8,
                                   colsample_bytree=.2,
                                   subsample=.8,
                                   reg_alpha=1e-3,
                               ),
                                   cr=dict(n_estimators=500, learning_rate=5e-2
                                           , bagging_fraction=0.1, bagging_freq=5)
                               )
                               )


def get_model(model_type, data_type, prediction_type):
    if prediction_type == 'company':
        model_parameters = company_model_parameters
    else:
        model_parameters = sector_model_parameters
    model_dict = dict(rf=RandomForestRegressor(**model_parameters['rf'][data_type], random_state=42),
                      svr=Pipeline([
                          ('scale', MinMaxScaler(feature_range=(-1, 1))),
                          ('regressor', svm.SVR(**model_parameters['svr'][data_type]))
                      ]),
                      lgb=Pipeline([
                          ('scale', MinMaxScaler(feature_range=(-1, 1))),
                          ('regressor',
                           lgb.LGBMRegressor(**model_parameters['lgb'][data_type], random_state=42, n_jobs=-1))
                      ])
                      )
    if model_type not in model_dict.keys():
        raise ValueError("Model not found in list")
    return model_dict[model_type]


def get_fit_regressor(x_train, y_cr_train, x_validation, y_validation, x_test, y_cr_test, data_type='cr',
                      model_type='rf', prediction_type='company', context=None,
                      columns=None,
                      get_cross_validation_results=True, suffix=None):
    if columns is not None:
        X_train, y_train = x_train[columns].copy(), y_cr_train.copy()
        X_validation, y_validation = x_validation[columns].copy(), y_validation.copy()
        X_test, y_test = x_test[columns].copy(), y_cr_test.copy()
    else:
        X_train, y_train = x_train.copy(), y_cr_train.copy()
        X_validation, y_validation = x_validation.copy(), y_validation.copy()
        X_test, y_test = x_test.copy(), y_cr_test.copy()

    print('train', X_train.shape, y_train.shape)
    print('validation', X_validation.shape, y_validation.shape)
    print('test', X_test.shape, y_test.shape)

    regressor = get_model(model_type=model_type, data_type=data_type, prediction_type=prediction_type)
    score = None
    if get_cross_validation_results:
        if x_validation is None or len(x_validation) == 0:
            raise ValueError("Should provide a validation set")
        x, y = pd.concat([X_train, X_validation]), pd.concat([y_train, y_validation])

        cv = TimeSeriesSplit(max_train_size=int(2 * len(x) / 3), n_splits=10)
        score = cross_validate(regressor, x.values, y.values.ravel(),
                               scoring=['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error'], n_jobs=-1,
                               verbose=0, cv=cv)

    regressor.fit(X_train.values, y_train.values.ravel())
    y_hat = regressor.predict(X_test.values)
    y_test['predicted'] = y_hat.reshape(-1, 1)

    y_hat_val = regressor.predict(X_validation.values)
    y_validation['predicted'] = y_hat_val.reshape(-1, 1)
    if suffix:
        y_test = y_test.add_suffix(suffix)
        y_validation = y_validation.add_suffix(suffix)

    print('validation', X_validation.shape, y_validation.shape)
    print('test', X_test.shape, y_test.shape)
    return regressor, y_validation, y_test, score
