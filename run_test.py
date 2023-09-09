import numpy as np
import pandas as pd
import time
import config as cfg
from sksurv.util import Surv
# import shap
from pycox.evaluation import EvalSurv
from xgbse.non_parametric import calculate_kaplan_vectorized
from sklearn.model_selection import train_test_split
from utility.survival import Survival
from tools.resume import Resume
from tools import regressors, feature_selectors
from utility.builder import Builder
from xgbse.metrics import approx_brier_score
from sksurv.metrics import concordance_index_censored
from tools.file_reader import FileReader
from tools.data_ETL import DataETL
from lifelines import WeibullAFTFitter
from lifelines import KaplanMeierFitter

import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

N_BOOT = cfg.N_BOOT
N_REPEATS = 1
NEW_DATASET = False
DATASET = "pronostia"
TYPE = "not_correlated"  # not_correlated
LINE_PLOT = 3
FEATURE_TO_SPLIT = "rms"
SPLIT_THRESHOLD = [] # [2] Only for xjtu
N_CONDITION = len (cfg.RAW_DATA_PATH_PRONOSTIA)
MERGE= False

def main():
    global BEARINGS
    global BOOT_NO
    global TEST_SIZE

    if DATASET == "pronostia":
        BEARINGS = 2
        BOOT_NO = cfg.N_BOOT_FOLD_UPSAMPLING
        TEST_SIZE = 0.5 
    elif DATASET == "xjtu":
        BEARINGS = 5
        BOOT_NO = cfg.N_BOOT_FOLD_UPSAMPLING
        TEST_SIZE = 0.3

    #For the first time running, a NEW_DATASET is needed
    if NEW_DATASET == True:
        Builder(DATASET).build_new_dataset(bootstrap=N_BOOT)

    #Prepare the object needed
    survival = Survival()
    data_util = DataETL(DATASET) 
    
    #Eventually plot and create a table for CV search 
    #Resumer.table_result_hyper()
    #Resumer.presentation(BEARINGS, BOOT_NO)

    #Extract information from the dataset selected from the config file
    cov_group = []
    boot_group = []
    info_group = []
    for i in range (0, N_CONDITION):
        cov, boot, info_pack = FileReader(DATASET).read_data(i)
        cov_group.append(cov)
        boot_group.append(boot)
        info_group.append(info_pack)

    #Transform information from the dataset selected from the config file
    data_container_X = []
    data_container_y= []
    if MERGE == True:
        data_X_merge = pd.DataFrame()
        for i, (cov, boot, info_pack) in enumerate(zip(cov_group, boot_group, info_group)):
            data_temp_X, deltaref_temp_y = data_util.make_surv_data_sklS(cov, boot, info_pack, N_BOOT, TYPE)
            if i== 0:
                deltaref_y_merge =  deltaref_temp_y
            else:
                deltaref_y_merge =  deltaref_y_merge.update(deltaref_temp_y)
            data_X_merge = pd.concat([data_X_merge, data_temp_X], ignore_index=True)
        data_container_X.append(data_X_merge)
        data_container_y.append(deltaref_y_merge)
    else:
        for i, (cov, boot, info_pack) in enumerate(zip(cov_group, boot_group, info_group)):
            data_temp_X, deltaref_y = data_util.make_surv_data_sklS(cov, boot, info_pack, N_BOOT, TYPE)
            data_container_X.append(data_temp_X)
            data_container_y.append(deltaref_y)

    #Load information from the dataset selected in the config file      
    for X, y in zip(data_container_X, data_container_y):

        #Information about the event estimation in event detector
        y_delta = y

        for n_repeat in range(N_REPEATS):
            
            #Indexing the dataset to avoid train/test leaking
            dummy_x= list(range(0, BEARINGS))
            dummy_y= list(range(0, BEARINGS))

            #Test/train split
            X_train, X_test, y_train, y_test = train_test_split(dummy_x, dummy_y, shuffle= False, test_size=TEST_SIZE, random_state=None)
            train_index = X_train
            test_index = X_test
            train = np.delete(X_train, slice(None))
            test = np.delete(X_test, slice(None))               
            
            #Load the indexed data  
            data_X_merge = pd.DataFrame()
            for element in train_index:
                data_X_merge_tr = pd.concat([data_X_merge, X [element]], ignore_index=True)
            for element in test_index:
                data_X_merge_te = pd.concat([data_X_merge, X [element]], ignore_index=True)

            data_X_train = data_X_merge_tr
            data_X_test = data_X_merge_te
            data_y_train = Surv.from_dataframe("Event", "Survival_time", data_X_train)
            data_y_test = Surv.from_dataframe("Event", "Survival_time", data_X_test)

            #Create an object for future plotting using test data
            Resumer = Resume(data_X_test, data_y_test, DATASET)

            S1, S2 = (data_X_train, data_y_train), (data_X_test, data_y_test)

            #Format and centering the data
            set_tr, set_te, set_tr_NN, set_te_NN = data_util.format_main_data(S1, S2)
            percent_ref = data_util.calculate_positions_percentages(set_te[0], FEATURE_TO_SPLIT, SPLIT_THRESHOLD)
            set_tr, set_te, set_tr_NN, set_te_NN = data_util.centering_main_data(set_tr, set_te, set_tr_NN, set_te_NN)
            val_ref = data_util.find_values_by_percentages(set_te[0], FEATURE_TO_SPLIT, percent_ref)

            #Set up a general format for general models and NNs
            X_train = set_tr[0]
            y_train = set_tr[1]
            X_test = set_te[0]
            y_test = set_te[1]
            X_train_NN = set_tr_NN[0]
            y_train_NN = set_tr_NN[1]
            X_test_NN = set_te_NN[0]
            y_test_NN = set_te_NN[1]

            #Eventually control the censored data
            #X_test_NN, y_test_NN = data_util.control_censored_data(X_test_NN, y_test_NN, percentage= 10)

            #Set event times for general models and only NNs
            lower, upper = np.percentile(S1[1]['Survival_time'], [10, 90])
            time_bins = np.arange(np.ceil(lower), np.floor(upper))
            lower_NN, upper_NN = np.percentile(y_test[y_test.dtype.names[1]], [10, 90])
            times = np.arange(np.ceil(lower_NN), np.floor(upper_NN))

            #Set up the models on test
            weibull_model = WeibullAFTFitter(alpha=0.4, penalizer=0.06)
            cph_model = regressors.CoxPH().make_model()
            rsf_model = regressors.RSF().make_model()
            boost_model = regressors.CoxBoost().make_model()
            NN_model = regressors.DeepSurv().make_model()
            NN_params = regressors.DeepSurv().get_best_hyperparams()
            DSM_model = regressors.DSM().make_model()
            DSM_params = regressors.DSM().get_best_hyperparams()

            #Set the feature selector and train/test split
            best_features = feature_selectors.NoneSelector(X_train, y_train, cph_model).get_features()
            X_train, X_test, X_train_NN, X_test_NN = X_train.loc[:, best_features], X_test.loc[:,best_features], X_train_NN.loc[:, best_features], X_test_NN.loc[:, best_features]

            #Format the data for NNs models
            x = X_train_NN.to_numpy()
            t = y_train_NN['time'].to_numpy()
            e = y_train_NN['event'].to_numpy()
            xte = X_test_NN.to_numpy()

            #Format the data for parametric models
            X_train_WB = pd.concat([X_train.reset_index(drop=True), pd.DataFrame(y_train['Survival_time'], columns=['Survival_time'])], axis=1)
            X_train_WB = pd.concat([X_train_WB.reset_index(drop=True), pd.DataFrame(y_train['Event'], columns=['Event'])], axis=1)
            X_test_WB = pd.concat([X_test.reset_index(drop=True), pd.DataFrame(y_test['Survival_time'], columns=['Survival_time'])], axis=1)
            X_test_WB = pd.concat([X_test_WB.reset_index(drop=True), pd.DataFrame(y_test['Event'], columns=['Event'])], axis=1)

            #Train the models and get the calculation time fro each
            start_time_weibull = time.time()
            weibull_model.fit(X_train_WB, duration_col='Survival_time', event_col='Event')
            end_time_weibull = time.time() - start_time_weibull
            start_time_cph = time.time()
            cph_model.fit(X_train, y_train)
            end_time_cph = time.time() - start_time_cph
            start_time_rsf = time.time()
            rsf_model.fit(X_train, y_train)
            end_time_rsf = time.time() - start_time_rsf
            start_time_boost = time.time()
            boost_model.fit(X_train, y_train)
            end_time_boost = time.time() - start_time_boost
            start_time_NN = time.time()
            NN_model.fit(x, t, e, vsize=0.3, **NN_params)
            end_time_NN = time.time() - start_time_NN
            start_time_DSM = time.time()
            DSM_model.fit(x, t, e, vsize=0.3, **DSM_params)
            end_time_DSM = time.time() - start_time_DSM

            #Set event times
            lower_NN_tr, upper_NN_tr = np.percentile(y_train[y_train.dtype.names[1]], [10, 90])
            times_tr = np.arange(np.ceil(lower_NN_tr), np.floor(upper_NN_tr))

            #Make the survival function for each model
            weibull_surv_func = survival.predict_survival_function(weibull_model, X_test_WB, time_bins)
            cph_surv_func = survival.predict_survival_function(cph_model, X_test, time_bins)
            boost_surv_func = survival.predict_survival_function(boost_model, X_test, time_bins)
            rsf_surv_func = survival.predict_survival_function(rsf_model, X_test, time_bins)
            NN_surv_func = survival.predict_survival_function(NN_model, xte, times)
            NN_surv_func_tr = survival.predict_survival_function(NN_model, x, times_tr)
            DSM_surv_func = survival.predict_survival_function(DSM_model, xte, times)
            DSM_surv_func_tr = survival.predict_survival_function(DSM_model, x, times_tr)

            #Make the hazard function for each model
            weibull_hazard_func = survival.predict_hazard_function(weibull_model, X_test_WB, time_bins)
            cph_hazard_func = survival.predict_hazard_function(cph_model, X_test, time_bins)
            boost_hazard_func = survival.predict_hazard_function(boost_model, X_test, time_bins)
            rsf_hazard_func = survival.predict_hazard_function(rsf_model, X_test, time_bins)

            km_mean, km_high, km_low = calculate_kaplan_vectorized(y_test['Event'].reshape(1, -1),
                                                                   y_test['Survival_time'].reshape(1, -1),
                                                                   time_bins)

            #Weibull C_INDEX
            weibull_c_index = np.mean(weibull_model.concordance_index_)
            ev = EvalSurv(weibull_surv_func.T, y_test['Survival_time'], y_test['Event'], censor_surv="km")
            weibull_c_index_td = ev.concordance_td()

            #CPH C_INDEX
            cph_c_index = concordance_index_censored(y_test['Event'], y_test['Survival_time'], cph_model.predict(X_test))[0]
            ev = EvalSurv(cph_surv_func.T, y_test['Survival_time'], y_test['Event'], censor_surv="km")
            cph_c_index_td = ev.concordance_td()

            #RSF C_INDEX
            rsf_c_index = concordance_index_censored(y_test['Event'], y_test['Survival_time'], rsf_model.predict(X_test))[0]
            ev = EvalSurv(rsf_surv_func.T, y_test['Survival_time'], y_test['Event'], censor_surv="km")
            rsf_c_index_td = ev.concordance_td()

            #Boost C_INDEX
            boost_c_index = concordance_index_censored(y_test['Event'], y_test['Survival_time'], boost_model.predict(X_test))[0]
            ev = EvalSurv(rsf_surv_func.T, y_test['Survival_time'], y_test['Event'], censor_surv="km")
            boost_c_index_td = ev.concordance_td()

            lower_NN, upper_NN = np.percentile(y_train[y_train.dtype.names[1]], [10, 90])
            times = np.arange(np.ceil(lower_NN), np.floor(upper_NN))

            risk_NN = NN_model.predict_risk(xte, t= times.tolist())
            risk_NN = [item[0] for item in risk_NN]

            #NN C_INDEX
            NN_c_index = concordance_index_censored(y_test['Event'], y_test['Survival_time'], risk_NN)[0]
        #        NN_c_index_tr = concordance_index_censored(y_train['Event'], y_train['Survival_time'], NN_surv_func_tr)[0]
            ev = EvalSurv(NN_surv_func.T, y_test['Survival_time'], y_test['Event'], censor_surv="km")
            NN_c_index_td = ev.concordance_td()

            #DSM C_INDEX
            risk_DSM = DSM_model.predict_risk(xte, t=list(times))
            risk_DSM = [item[0] for item in risk_DSM]
            DSM_c_index = concordance_index_censored(y_test['Event'], y_test['Survival_time'], risk_DSM)[0]
        #        DSM_c_index_tr = concordance_index_censored(y_train['Event'], y_train['Survival_time'], DSM_surv_func_tr)[0]
            ev = EvalSurv(DSM_surv_func.T, y_test['Survival_time'], y_test['Event'], censor_surv="km")
            DSM_c_index_td = ev.concordance_td()

            #Weibull BS
            weibull_bs = approx_brier_score(y_test, weibull_surv_func)

            #CPH BS
            cph_surv_probs = pd.DataFrame(cph_surv_func)
            cph_bs = approx_brier_score(y_test, cph_surv_probs)

            #RSF BS
            rsf_surv_probs = pd.DataFrame(boost_surv_func)
            rsf_bs = approx_brier_score(y_test, rsf_surv_probs)

            #Boost BS
            boost_surv_probs = pd.DataFrame(rsf_surv_func)
            boost_bs = approx_brier_score(y_test, boost_surv_probs)

            #NN BS
            NN_surv_probs = NN_model.predict_survival(xte)
            NN_bs = approx_brier_score(y_test, NN_surv_probs)

            NN_surv_probs_tr = NN_model.predict_survival(x)
            NN_bs_tr = approx_brier_score(y_train, NN_surv_probs_tr)

            #DSM BS
            times = np.arange(np.ceil(lower_NN), np.floor(upper_NN))
            DSM_surv_probs = pd.DataFrame(DSM_model.predict_survival(xte, t= list(times)))
            DSM_bs = approx_brier_score(y_test, DSM_surv_probs)
            DSM_surv_probs_tr = pd.DataFrame(NN_model.predict_survival(x, t=list(times_tr)))
            DSM_bs_tr = approx_brier_score(y_train, DSM_surv_probs_tr)

            #Save and show results
            df_CI = pd.DataFrame(columns=["Model", "CI score"])
            df_B = pd.DataFrame(columns=["Model", "Brier score"])

            print("Performance info: ")
            print(f"Weibull model: TrTime, CI, CItd, BS - {end_time_weibull}/{weibull_c_index}/{weibull_c_index_td}/{weibull_bs}")
            df_CI, df_B = Resumer.plot_performance(False, df_CI, df_B, "Weibull AFT", weibull_c_index, weibull_bs)
            print(f"CPH model: TrTime, CI, CItd, BS - {end_time_cph}/{cph_c_index}/{cph_c_index_td}/{cph_bs}")
            df_CI, df_B = Resumer.plot_performance(False, df_CI, df_B, "Cox PH", cph_c_index, cph_bs)
            print(f"RSF model: TrTime, CI, CItd, BS - {end_time_rsf}/{rsf_c_index}/{rsf_c_index_td}/{rsf_bs}")
            df_CI, df_B = Resumer.plot_performance(False, df_CI, df_B, "Random Survival Forest", rsf_c_index, rsf_bs)
            print(f"Boost model: TrTime, CI, CItd, BS - {end_time_boost}/{boost_c_index}/{boost_c_index_td}/{boost_bs}")
            df_CI, df_B = Resumer.plot_performance(False, df_CI, df_B, "Gradient Boosting", boost_c_index, boost_bs)
            print(f"NN model: TrTime, CI, CItd, BS - {end_time_NN}/{NN_c_index}/{NN_c_index_td}/{NN_bs}")
            df_CI, df_B = Resumer.plot_performance(False, df_CI, df_B, "Neural Network", NN_c_index, NN_bs)
            print(f"DSM model: TrTime, CI, CItd, BS - {end_time_DSM}/{DSM_c_index}/{DSM_c_index_td}/{DSM_bs}")
            df_CI, df_B = Resumer.plot_performance(False, df_CI, df_B, "DSM", DSM_c_index, DSM_bs)

        #        print ("Overfitting - underfitting info: ")
        #        print(f"NN model train: TrCI, TrBS - {NN_bs}/{NN_bs_tr}")
        #        print(f"DSM model train: TrCI, TrBS - {DSM_bs}/{DSM_bs_tr}")

            print("Latex performance table: ")
            print(f"CoxPH & {round(end_time_cph, 3)} & {round(cph_c_index, 3)} & {round(cph_c_index_td, 3)} & {round(cph_bs, 3)} \\\\" +
                  f"RSF & {round(end_time_rsf, 3)} & {round(rsf_c_index, 3)} & {round(rsf_c_index_td, 3)} & {round(rsf_bs, 3)} \\\\" +
                  f"CoxBoost & {round(end_time_boost, 3)} & {round(boost_c_index, 3)} & {round(boost_c_index_td, 3)} & {round(boost_bs, 3)}\\\\" +
                  f"DeepSurv & {round(end_time_NN, 3)} & {round(NN_c_index, 3)} & {round(NN_c_index_td, 3)} & {round(NN_bs, 3)}\\\\" +
                  f"DSM & {round(end_time_DSM, 3)} & {round(DSM_c_index, 3)} & {round(DSM_c_index_td, 3)} & {round(DSM_bs, 3)}\\\\" +
                  f"WeibullAFT & {round(end_time_weibull, 3)} & {round(weibull_c_index, 3)} & {round(weibull_c_index_td, 3)} & {round(weibull_bs, 3)}\\\\")
            
            print ("The event detector establish an event for each bearing in: ", y_delta)

            #Plotting aggregate survival lines and grouping
            Km = KaplanMeierFitter()
            Km.fit(durations= X_test_WB["Survival_time"], event_observed= X_test_WB["Survival_time"])
            Km.predict(time_bins)
            surv_label = []
            surv_label.append(weibull_surv_func)
            surv_label.append(cph_surv_func)
            surv_label.append(boost_surv_func)
            surv_label.append(rsf_surv_func)
            surv_label.append(NN_surv_probs)
            surv_label.append(DSM_surv_probs)
            Resumer.plot_aggregate_sl(Km, surv_label)

            # # Make SHAP values
            # if (n_repeat== N_REPEATS - 1):

            #     explainer_CPH = shap.Explainer(cph_model.predict, X_test)
            #     shap_values_CPH = explainer_CPH(X_test)

            #     explainer_RSF = shap.Explainer(rsf_model.predict, X_test)
            #     shap_values_RSF = explainer_RSF(X_test)

            #     explainer_B = shap.Explainer(boost_model.predict, X_test)
            #     shap_values_B = explainer_B(X_test)

            #     explainer_NN = shap.Explainer(NN_model.predict_survival, X_test_NN)
            #     shap_values_NN = explainer_NN(X_test)

        #     Resumer.plot_simple_sl (y_test, weibull_surv_func, "Weibull AFT")

        #     Resumer.plot_simple_sl (y_test, cph_surv_func, "Cox PH")
        #     Resumer.plot_sl_ci (y_test, cph_surv_func, "Cox PH")
        #     # Resumer.plot_shap(explainer_CPH, shap_values_CPH, X_test, "Cox PH")

        #     Resumer.plot_simple_sl (y_test, rsf_surv_func, "Random Survival Forest")
        #     Resumer.plot_sl_ci (y_test, rsf_surv_func, "Random Survival Forest")
        #     # Resumer.plot_shap(explainer_RSF, shap_values_RSF, X_test, "Random Survival Forest")

        #     Resumer.plot_simple_sl (y_test, boost_surv_func, "Gradient Boosting")
        #     Resumer.plot_sl_ci (y_test, boost_surv_func, "Gradient Boosting")
        #     # Resumer.plot_shap(explainer_B, shap_values_B, X_test, "Gradient Boosting")

        #     NN_surv_func = pd.DataFrame(NN_surv_func)
        #     Resumer.plot_simple_sl (y_test, NN_surv_func, "Neural Network")
        # #    Resumer.plot_shap(explainer_NN, shap_values_NN, X_test, "Neural Network")

        #    df_CI, df_B = Resumer.plot_performance(True, df_CI, df_B)

if __name__ == "__main__":
    main()
