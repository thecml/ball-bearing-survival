PARAMS_CPH = {'alpha': 0.1,
            'ties': 'breslow',
            'n_iter': 100,
            'tol': 0.1}

PARAMS_CPH_RIDGE = {'alpha': 0.5,
                    'ties': 'breslow',
                    'n_iter': 100,
                    'tol': 1e-9}

PARAMS_CPH_LASSO = {'l1_ratio': 1.0,
                    'alpha_min_ratio': 0.5,
                    'fit_baseline_model': True,
                    'normalize': False,
                    'tol': 1e-7,
                    'max_iter': 100000}

PARAMS_CPH_ELASTIC = {'l1_ratio': 0.5,
                    'alpha_min_ratio': 0.1,
                    'fit_baseline_model': True,
                    'normalize': False,
                    'tol': 1e-7,
                    'max_iter': 100000}

PARAMS_RSF = {
    'n_estimators': 100,
    'max_depth' : 7,
    'min_samples_split': 10,
    'min_samples_leaf': 2,
    'max_features': None,
    'random_state': 0
}

PARAMS_GRADBOOST = {'n_estimators': 400,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'loss': 'coxph',
                    'min_samples_split': 5,
                    'min_samples_leaf': 4,
                    'max_features': None,
                    'dropout_rate': 0.0,
                    'subsample': 1.0,
                    'random_state': 0}

PARAMS_GRADBOOST_DART = {'n_estimators': 100,
                        'learning_rate': 0.1,
                        'max_depth': 3,
                        'loss': 'coxph',
                        'min_samples_split': 2,
                        'min_samples_leaf': 1,
                        'max_features': None,
                        'dropout_rate': 0.2,
                        'subsample': 1.0,
                        'random_state': 0}

PARAMS_SVM = {'alpha': 1, 
              'rank_ratio': 0.8,
              'max_iter': 40, 
              'optimizer': 'avltree'}

PARAMS_DEEPSURV = {'batch_size' : 16,
                   'learning_rate' : 0.01,
                   'iters': 50}

PARAMS_DSM = {'batch_size' : 32,
                   'learning_rate' : 0.01,
                   'iters': 100}

PARAMS_WEIBULL = {'alpha': 0.2,
                'penalizer': 0.02,
                'l1_ratio': 0.0,
                'fit_intercept': True}
