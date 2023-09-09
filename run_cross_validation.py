import numpy as np
import pandas as pd
from time import time
import math
import argparse
import warnings
import config as cfg
import re
from pycox.evaluation import EvalSurv
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from xgbse.metrics import approx_brier_score
from sklearn.model_selection import RandomizedSearchCV
from tools.feature_selectors import NoneSelector, LowVar, SelectKBest4, SelectKBest8, RegMRMR4, RegMRMR8, UMAP8, VIF4, VIF8, PHSelector
from tools.regressors import CoxPH, CphRidge, CphLASSO, CphElastic, RSF, CoxBoost, GradientBoostingDART, WeibullAFT, LogNormalAFT, LogLogisticAFT, DeepSurv, DSM
from tools.file_reader import FileReader
from tools.data_ETL import DataETL
from utility.builder import Builder
from tools.experiments import SurvivalRegressionCV
from utility.survival import Survival
from auton_survival import DeepCoxPH
from auton_survival import DeepSurvivalMachines
from lifelines import WeibullAFTFitter
from sklearn.model_selection import ParameterSampler

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

N_BOOT = 3
PLOT = True
RESUME = True
NEW_DATASET = True
N_REPEATS = 1
N_ITER = 10
N_SPLITS = 5

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        required=False,
                        default=None)
    parser.add_argument('--typedata', type=str,
                        required=False,
                        default=None)
    parser.add_argument('--merge', type=str,
                        required=False,
                        default=None)
    args = parser.parse_args()

    global DATASET
    global TYPE
    global MERGE
    global N_CONDITION
    global N_BEARING
    global N_SPLITS
    global TRAIN_SIZE

    if args.dataset:
        DATASET = args.dataset

    if args.typedata:
        TYPE = args.typedata
    
    if args.merge:
        MERGE = args.merge
    
    DATASET = "pronostia"
    TYPE = "correlated"
    MERGE = False

    if DATASET == "xjtu":
        N_CONDITION = len(cfg.RAW_DATA_PATH_XJTU)
        N_BEARING = cfg.N_REAL_BEARING_XJTU
        TRAIN_SIZE = 0.7
    elif DATASET == "pronostia":
        N_CONDITION = len(cfg.RAW_DATA_PATH_PRONOSTIA)
        N_BEARING = cfg.N_REAL_BEARING_PRONOSTIA
        TRAIN_SIZE = 0.5
    
    if NEW_DATASET== True:
        Builder(DATASET).build_new_dataset(bootstrap=N_BOOT)
    
    models = [DSM]
    survival = Survival()
    
    cov_group = []
    boot_group = []
    info_group = []
    for i in range (0, N_CONDITION):
        cov, boot, info_pack = FileReader(DATASET).read_data(i)
        cov_group.append(cov)
        boot_group.append(boot)
        info_group.append(info_pack)

    data_container_X = []
    data_container_y= []
    if MERGE == True:
        data_X_merge = pd.DataFrame()
        for i, (cov, boot, info_pack) in enumerate(zip(cov_group, boot_group, info_group)):
            data_temp_X, deltaref_temp_y = DataETL(DATASET).make_surv_data_sklS(cov, boot, info_pack, N_BOOT, TYPE)
            if i== 0:
                deltaref_y_merge =  deltaref_temp_y
            else:
                deltaref_y_merge =  deltaref_y_merge.update(deltaref_temp_y)
            data_X_merge = pd.concat([data_X_merge, data_temp_X], ignore_index=True)
        data_container_X.append(data_X_merge)
        data_container_y.append(deltaref_y_merge)
    else:
        for i, (cov, boot, info_pack) in enumerate(zip(cov_group, boot_group, info_group)):
            data_temp_X, deltaref_y = DataETL(DATASET).make_surv_data_sklS(cov, boot, info_pack, N_BOOT, TYPE)
            data_container_X.append(data_temp_X)
            data_container_y.append(deltaref_y)
                                                                          
    for i, (data_X, data_y) in enumerate(zip(data_container_X, data_container_y)):

        dummy_x = list(range(0, int(np.floor(N_BEARING * TRAIN_SIZE)), 1))
        data_index = dummy_x
                        
        data_X_merge = pd.DataFrame()
        for element in data_index:
            data_X_merge = pd.concat([data_X_merge, data_X[element]], ignore_index=True)

        data_X = data_X_merge
        data_y = Surv.from_dataframe("Event", "Survival_time", data_X)

        T1 = (data_X, data_y)

        print(f"Started evaluation of {len(models)} models. Dataset: {DATASET}. Type: {TYPE}")
        for model_builder in models:
            model_name = model_builder.__name__
            
            # Set parameter space
            space = model_builder().get_tuneable_params()
            param_list = list(ParameterSampler(space, n_iter=N_ITER, random_state=0))
            
            model_results = pd.DataFrame()
            for sample in param_list:
                param_results = pd.DataFrame()
                kf = KFold(n_splits=N_SPLITS, random_state=0, shuffle=True)
                for split_idx, (train, test) in enumerate(kf.split(data_X, data_y)):
                    # Prepare data
                    ti, cvi, ti_NN, cvi_NN = DataETL(DATASET).format_main_data_Kfold(T1, train, test)
                    ti, cvi, ti_NN, cvi_NN = DataETL(DATASET).centering_main_data(ti, cvi, ti_NN, cvi_NN)
                    
                    # Set event times
                    lower, upper = np.percentile(ti[1][ti[1].dtype.names[1]], [10, 90])
                    times = np.arange(math.ceil(lower), math.floor(upper)).tolist()

                    # Find hyperparams via CV
                    if model_name == 'WeibullAFT':
                        x_ti_wf = pd.concat([ti[0].reset_index(drop=True), pd.DataFrame(ti[1]['Survival_time'], columns=['Survival_time'])], axis=1)
                        x_ti_wf = pd.concat([x_ti_wf.reset_index(drop=True), pd.DataFrame(ti[1]['Event'], columns=['Event'])], axis=1)
                        x_cvi_wf = pd.concat([cvi[0].reset_index(drop=True), pd.DataFrame(cvi[1]['Survival_time'], columns=['Survival_time'])], axis=1)
                        x_cvi_wf = pd.concat([x_cvi_wf.reset_index(drop=True), pd.DataFrame(cvi[1]['Event'], columns=['Event'])], axis=1)
                        model = WeibullAFTFitter(**sample)
                        model.fit(x_ti_wf, duration_col='Survival_time', event_col='Event')
                        preds = survival.predict_survival_function(model, x_cvi_wf, times)
                        ev = EvalSurv(preds.T, cvi[1]['Survival_time'], cvi[1]['Event'], censor_surv="km")
                        c_index = ev.concordance_td()
                        res_sr = pd.Series([str(model_name), split_idx, sample, c_index],
                                           index=["ModelName", "SplitIdx", "Params", "CIndex"])
                        param_results = pd.concat([param_results, res_sr.to_frame().T], ignore_index=True)
                    elif model_name == "DeepSurv":
                        x = ti_NN[0].to_numpy()
                        t = ti_NN[1].loc[:, "time"].to_numpy()
                        e = ti_NN[1].loc[:, "event"].to_numpy()
                        model = DeepCoxPH(layers=[32, 32])
                        model.fit(x, t, e, vsize=0.3, **sample)
                        xte = cvi_NN[0].to_numpy()
                        preds = survival.predict_survival_function(model, xte, times)
                        ev = EvalSurv(preds.T, cvi[1]['Survival_time'], cvi[1]['Event'], censor_surv="km")
                        c_index = ev.concordance_td()
                        res_sr = pd.Series([str(model_name), split_idx, sample, c_index],
                                           index=["ModelName", "SplitIdx", "Params", "CIndex"])
                        param_results = pd.concat([param_results, res_sr.to_frame().T], ignore_index=True)
                    elif model_name == "DSM":
                        x = ti_NN[0].to_numpy()
                        t = ti_NN[1].loc[:, "time"].to_numpy()
                        e = ti_NN[1].loc[:, "event"].to_numpy()
                        model = DeepSurvivalMachines(layers=[32, 32])
                        model.fit(x, t, e, vsize=0.3, **sample)
                        xte = cvi_NN[0].to_numpy()
                        preds = survival.predict_survival_function(model, xte, list(times))
                        ev = EvalSurv(preds.T, cvi[1]['Survival_time'], cvi[1]['Event'], censor_surv="km")
                        c_index = ev.concordance_td()
                        res_sr = pd.Series([str(model_name), split_idx, sample, c_index],
                                        index=["ModelName", "SplitIdx", "Params", "CIndex"])
                        param_results = pd.concat([param_results, res_sr.to_frame().T], ignore_index=True)
                    else:
                        model = model_builder().get_estimator()
                        model.set_params(**sample)
                        model.fit(ti[0], ti[1])
                        preds = survival.predict_survival_function(model, cvi[0], times)
                        ev = EvalSurv(preds.T, cvi[1]['Survival_time'], cvi[1]['Event'], censor_surv="km")
                        c_index = ev.concordance_td()
                        res_sr = pd.Series([str(model_name), split_idx, sample, c_index],
                                           index=["ModelName", "SplitIdx", "Params", "CIndex"])
                        param_results = pd.concat([param_results, res_sr.to_frame().T], ignore_index=True)
                mean_c_index = param_results['CIndex'].mean()
                res_sr = pd.Series([str(model_name), sample, mean_c_index],
                                   index=["ModelName", "Params", "CIndex"])
                model_results = pd.concat([model_results, res_sr.to_frame().T], ignore_index=True)
            model_results.to_csv(f"data/logs/{DATASET}/{TYPE}/" + f"{model_name}_cv_results.csv")
        
if __name__ == "__main__":
    main()
