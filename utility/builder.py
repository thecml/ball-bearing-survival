import os
import pandas as pd
import re
from tools.featuring import Featuring
import config as cfg


class Builder:

    def __init__ (self, dataset):
        if dataset == "xjtu":
            self.total_bearings= cfg.N_BEARING_TOT_XJTU
            self.bootstrapped_fold= cfg.N_BOOT_FOLD_XJTU
            self.raw_main_path= cfg.RAW_DATA_PATH_XJTU
            self.aggregate_main_path= cfg.DATASET_PATH_XJTU
            self.real_bearing= cfg.N_REAL_BEARING_XJTU
        elif dataset == "pronostia":
            self.total_bearings= cfg.N_BEARING_TOT_PRONOSTIA
            self.bootstrapped_fold= cfg.N_BOOT_FOLD_PRONOSTIA
            self.raw_main_path= cfg.RAW_DATA_PATH_PRONOSTIA
            self.aggregate_main_path= cfg.DATASET_PATH_PRONOSTIA
            self.real_bearing= cfg.N_REAL_BEARING_PRONOSTIA
        self.dataset= dataset 

    def build_new_dataset (self,bootstrap= 0):            
        self.from_raw_to_csv(bootstrap)
        self.aggregate_and_refine()

    def from_raw_to_csv (self, bootno):
        for z, group in enumerate(self.raw_main_path):
            j = 1
            for bearing in range (1, self.real_bearing + 1, 1):
                dataset_path = group + "Bearing1_" + str(bearing)
                if self.dataset == "xjtu":
                    datasets, bootstrap_val = Featuring().time_features_xjtu(dataset_path, bootstrap= bootno)
                elif self.dataset == "pronostia": 
                    datasets, bootstrap_val = Featuring().time_features_pronostia(dataset_path, bootstrap= bootno)

                i = 1
                for dataset in datasets:
                    dataset.columns= ['B' + str(j) + '_mean' , 'B' + str(j) + '_std', 'B' + str(j) + '_skew', 'B' + str(j) + '_kurtosis', 'B' + str(j) + '_entropy', 
                                    'B' + str(j) + '_rms','B' + str(j) + '_max', 'B' + str(j) + '_p2p', 'B' + str(j) + '_crest', 'B' + str(j) + '_clearence', 
                                    'B' + str(j) + '_shape', 'B' + str(j) + '_impulse', 'B' + str(j) + '_FoH', 'B' + str(j) + '_FiH', 'B' + str(j) + '_FrH', 
                                    'B' + str(j) + '_FrpH', 'B' + str(j) + '_FcaH', 'B' + str(j) + '_Fo', 'B' + str(j) + '_Fi', 'B' + str(j) + '_Fr', 'B' + str(j) + '_Frp', 'B' + str(j) + '_Fca',
                                    'B' + str(j) + '_noise', 'B' + str(j) + '_Event', 'B' + str(j) + '_Survival_time',
                                    'B' + str(j + 1) + '_mean', 'B' + str(j + 1) + '_std', 'B' + str(j + 1) + '_skew', 'B' + str(j + 1) + '_kurtosis', 'B' + str(j + 1) + '_entropy', 
                                    'B' + str(j + 1) +'_rms', 'B' + str(j + 1) + '_max', 'B' + str(j + 1) + '_p2p', 'B' + str(j + 1) + '_crest' , 'B' + str(j + 1) + '_clearence', 
                                    'B' + str(j + 1) + '_shape','B' + str(j + 1) + '_impulse', 'B' + str(j + 1) + '_FoH', 'B' + str(j + 1) + '_FiH', 'B' + str(j + 1) + '_FrH', 
                                    'B' + str(j + 1) + '_FrpH', 'B' + str(j + 1) + '_FcaH', 'B' + str(j + 1) + '_Fo', 'B' + str(j + 1) + '_Fi', 'B' + str(j + 1) + '_Fr', 'B' + str(j + 1) + '_Frp', 'B' + str(j + 1) + '_Fca', 
                                    'B' + str(j + 1) + '_noise', 'B' + str(j + 1) + '_Event', 'B' + str(j + 1) + '_Survival_time']
                        
                    dataname= self.aggregate_main_path + "Bearing1_" + str(bearing) + "_" + str(i) + "_timefeature" + "_" + str(z) + ".csv"
                    dataset.to_csv(dataname, index= False)
                    i += 1
                    j += 2

                dataname= self.aggregate_main_path + "Bearing1_" + str(bearing) + "_bootstrap" + "_" + str(z) + ".csv"
                bootstrap_val.to_csv(dataname, index= False)

    def aggregate_and_refine (self):
        set_analytic= pd.DataFrame()
        set_covariates= pd.DataFrame()
        set_boot= pd.DataFrame()
        for i, group in enumerate(self.raw_main_path):
            for filename in os.listdir(self.aggregate_main_path):
                if re.search('^Bearing.*timefeature_' + str(i), filename):          
                    # datafile = pd.read_csv(os.path.join(self.aggregate_main_path, filename))
                    # set_analytic_aux= datafile.iloc[:, 12: 17]
                    # set_cov_aux= datafile.iloc[:, 0: 12]
                    # set_analytic= pd.concat([set_analytic, set_analytic_aux], axis= 1)
                    # set_covariates= pd.concat([set_covariates, set_cov_aux], axis= 1)
                    # set_cov_aux= datafile.iloc[:, 17: 19]     
                    # set_covariates= pd.concat([set_covariates, set_cov_aux], axis= 1)   

                    # set_analytic_aux= datafile.iloc[:, 31: 36]
                    # set_cov_aux= datafile.iloc[:, 19: 31]
                    # set_analytic= pd.concat([set_analytic, set_analytic_aux], axis= 1)
                    # set_covariates= pd.concat([set_covariates, set_cov_aux], axis= 1)
                    # set_cov_aux= datafile.iloc[:, 36: 38]     
                    # set_covariates= pd.concat([set_covariates, set_cov_aux], axis= 1)

                    # datafile = pd.read_csv(os.path.join(self.aggregate_main_path, filename))
                    # set_analytic_aux= datafile.iloc[:, 12: 17]
                    # set_cov_aux= datafile.iloc[:, 0: 17]
                    # set_analytic= pd.concat([set_analytic, set_analytic_aux], axis= 1)
                    # set_covariates= pd.concat([set_covariates, set_cov_aux], axis= 1)
                    # set_cov_aux= datafile.iloc[:, 17: 19]     
                    # set_covariates= pd.concat([set_covariates, set_cov_aux], axis= 1)   

                    # set_analytic_aux= datafile.iloc[:, 31: 36]
                    # set_cov_aux= datafile.iloc[:, 19: 36]
                    # set_analytic= pd.concat([set_analytic, set_analytic_aux], axis= 1)
                    # set_covariates= pd.concat([set_covariates, set_cov_aux], axis= 1)
                    # set_cov_aux= datafile.iloc[:, 36: 38]     
                    # set_covariates= pd.concat([set_covariates, set_cov_aux], axis= 1)

                    datafile = pd.read_csv(os.path.join(self.aggregate_main_path, filename))
                    set_analytic_aux= datafile.iloc[:, 12: 17]
                    set_cov_aux= datafile.iloc[:, 0: 23]
                    set_analytic= pd.concat([set_analytic, set_analytic_aux], axis= 1)
                    set_covariates= pd.concat([set_covariates, set_cov_aux], axis= 1)
                    set_cov_aux= datafile.iloc[:, 23: 25]     
                    set_covariates= pd.concat([set_covariates, set_cov_aux], axis= 1)   

                    set_analytic_aux= datafile.iloc[:, 37: 42]
                    set_cov_aux= datafile.iloc[:, 25: 48]
                    set_analytic= pd.concat([set_analytic, set_analytic_aux], axis= 1)
                    set_covariates= pd.concat([set_covariates, set_cov_aux], axis= 1)
                    set_cov_aux= datafile.iloc[:, 48: 51]     
                    set_covariates= pd.concat([set_covariates, set_cov_aux], axis= 1)

                elif re.search('^Bearing.*bootstrap_' + str(i), filename):
                    col_label = re.findall("_\d", filename)[0][1]      
                    datafile = pd.read_csv(os.path.join(self.aggregate_main_path, filename), index_col=False)
                    datafile.rename(columns = {'Bootstrap values': col_label}, inplace = True)
                    set_boot = pd.concat([set_boot, datafile], axis= 1)       

            set_analytic.to_csv(self.aggregate_main_path + 'analytic_' + str(i) + '.csv', index= False)
            set_covariates.to_csv(self.aggregate_main_path + 'covariates_'  + str(i) + '.csv', index= False)
            set_boot.to_csv(self.aggregate_main_path + 'boot_' + str(i) + '.csv', index= False)

            set_analytic= pd.DataFrame()
            set_covariates= pd.DataFrame()
            set_boot= pd.DataFrame()