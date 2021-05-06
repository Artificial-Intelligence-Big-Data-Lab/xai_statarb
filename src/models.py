import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_validate

model_parameters = dict(rf=dict(cr=dict(n_estimators=500, min_samples_leaf=5, max_features=1, oob_score=True),
                                ti=dict(n_estimators=150, max_depth=12, max_features=1, oob_score=True)))


def get_fit_regressor(x_train, y_cr_train, x_validation, y_validation, x_test, y_cr_test, data_type='cr', context=None,
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

    regressor = RandomForestRegressor(**model_parameters['rf'][data_type], random_state=42)
    # regressor = Pipeline([('scaler', MinMaxScaler(feature_range=(-1, 1)))
    #                          , ('svc', linear_model.TweedieRegressor( alpha=0.001))])
    # # regressor = ExtraTreesRegressor(n_estimators=350, max_samples=0.4, max_features=1, oob_score=True, bootstrap=True, random_state=42)
    # regressor.fit(X,y.values.ravel())
    # # save_path= './LIME/models/{0}_cr_{1}_{2}_{3}.joblib'.format(context["ticker"],context["method"],context["start"].strftime("%Y-%m-%d"),context["end"].strftime("%Y-%m-%d"))
    # # joblib.dump(regressor,save_path)

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
