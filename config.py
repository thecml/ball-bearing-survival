#N_BEARING_TOT_XJTU= 50
#N_REAL_BEARING_XJTU= 5
N_BOOT= 3
N_BOOT_FOLD_XJTU= (2 + 3) * 2
N_BOOT_FOLD_UPSAMPLING= N_BOOT_FOLD_XJTU * 20
N_SIGNALS_XJTU= 2 + 1
N_BOOT_FOLD_PRONOSTIA= (2 + 3) * 2
N_SIGNALS_PRONOSTIA= 2 + 1
DATASET_PATH_XJTU= "./data/XJTU-SY/csv/"
RAW_DATA_PATH_XJTU= ["./data/XJTU-SY/35Hz12kN/"] #37.5Hz11kN # 40Hz10kN ------------  , "./data/XJTU-SY/37.5Hz11kN/", "./data/XJTU-SY/40Hz10kN/"
N_REAL_BEARING_XJTU= 5
N_BEARING_TOT_XJTU= N_REAL_BEARING_XJTU * N_BOOT_FOLD_XJTU
DATASET_PATH_PRONOSTIA= "./data/PRONOSTIA/csv/"
RAW_DATA_PATH_PRONOSTIA= ["./data/PRONOSTIA/25Hz5kN/"]  #25Hz5kN # 27.65Hz4.2kN   ---------- , "./data/PRONOSTIA/25Hz5kN/", "./data/PRONOSTIA/27.65Hz4.2kN/", "./data/PRONOSTIA/30Hz4kN/"
N_REAL_BEARING_PRONOSTIA= 2
N_BEARING_TOT_PRONOSTIA= N_REAL_BEARING_PRONOSTIA * N_BOOT_FOLD_PRONOSTIA
RESULT_PATH_XJTU= "./data/XJTU-SY/results/"
SAMPLE_PATH_XJTU= "./data/XJTU-SY/csv/"
RESULT_PATH_PRONOSTIA= "./data/PRONOSTIA/results/"
SAMPLE_PATH_PRONOSTIA= "./data/PRONOSTIA/csv/"
HYPER_RESULTS= "./logs/"

FREQUENCY_BANDS1 = {'xjtu_start': [10, 32, 69, 105, 169],
                       'xjtu_stop': [16, 38, 75, 111, 175]}
FREQUENCY_BANDS2 = {'xjtu_start': [11, 34, 74, 112, 181],
                       'xjtu_stop': [17, 40, 80, 118, 187]}
FREQUENCY_BANDS3 = {'xjtu_start': [12, 37, 80, 120, 193],
                       'xjtu_stop': [17, 43, 86, 126, 199]}
FREQUENCY_BANDS4 = {'pronostia_start': [7, 22, 87, 137, 181],
                       'pronostia_stop': [13, 28, 93, 143, 187]} 
FREQUENCY_BANDS5 = {'pronostia_start': [9, 25, 96, 151, 201],
                       'pronostia_stop': [15, 31, 102, 157, 207]}    
FREQUENCY_BANDS6 = {'pronostia_start': [10, 27, 105, 165, 214],
                       'pronostia_stop': [16, 33, 111, 171, 220]}
   
PH_EXCLUSION = {'pronostia_not_corr': ['Fca','Fi','Fo','Fr','Frp', 'rms', 'clearence', 'FcaH', 'FiH', 'kurtosis', 'FoH', 'entropy', 'impulse', 'mean'],
                'pronostia_corr': ['Fca','Fi','Fo','Fr','Frp','noise', 'mean', 'std', 'kurtosis', 'rms', 'entropy', 'FrH', 'FoH', 'FrpH'],
                'pronostia_boot': ['Fca','Fi','Fo','Fr','Frp', 'mean', 'skew', 'FcaH', 'FiH', 'FoH', 'FrH', 'std', 'kurtosis', 'entropy', 'rms'],
                'xjtu_not_corr': ['Fca','Fi','Fo','Fr','Frp', 'kurtosis', 'FoH', 'rms', 'FcaH', 'noise', 'FiH', 'max', 'p2p'],
                'xjtu_corr': ['Fca','Fi','Fo','Fr','Frp','mean', 'std', 'skew', 'FcaH', 'clearence', 'max', 'kurtosis', 'shape', 'crest'],
                'xjtu_boot': ['Fca','Fi','Fo','Fr','Frp','entropy', 'FrpH', 'FcaH']}  

PARAMS_CPH = {'alpha': 0.1,
              'ties': 'breslow',
              'n_iter': 50,
              'tol': 1e-1}

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

PARAMS_RSF = {'n_estimators': 100,
              'max_depth' : 7,
              'min_samples_split': 2,
              'min_samples_leaf': 4,
              'max_features': None,
              'random_state': 0}

PARAMS_GRADBOOST = {'n_estimators': 400,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'loss': 'coxph',
                    'min_samples_split': 2,
                    'min_samples_leaf': 4,
                    'max_features': None,
                    'dropout_rate': 0,
                    'subsample': 1.0,
                    'random_state': 0}

PARAMS_GRADBOOST_DART = {'n_estimators': 200,
                        'learning_rate': 0.05,
                        'max_depth': 5,
                        'loss': 'coxph',
                        'min_samples_split': 10,
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
                   'learning_rate' : 1e-3,
                   'iters': 10}

PARAMS_DSM = {'batch_size' :16,
              'learning_rate' : 1e-2,
              'iters': 100}

PARAMS_WEIBULL = {'alpha': 0.4,
                  'penalizer': 0.04,
                  'l1_ratio': 0.0,
                  'fit_intercept': True}