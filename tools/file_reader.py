import pickle
from pathlib import Path
import pandas as pd
from utility.event import Event
import config as cfg

class FileReader:

    def __init__ (self, dataset):
        if (dataset == "xjtu"):
            self.dataset_path = cfg.DATASET_PATH_XJTU
        elif (dataset == "pronostia"):
            self.dataset_path = cfg.DATASET_PATH_PRONOSTIA
        self.dataset= dataset

    def read_data_kaggle (self):
        set1 = pd.read_csv("src/dataset/set1_timefeatures.csv")
        set2 = pd.read_csv("src/dataset/set2_timefeatures.csv")
        set3 = pd.read_csv("src/dataset/set3_timefeatures.csv")
        set1 = set1.rename(columns={'Unnamed: 0':'time'})
        set1.set_index('time')
        set2 = set2.rename(columns={'Unnamed: 0':'time'})
        set2.set_index('time')
        set3 = set3.rename(columns={'Unnamed: 0':'time'})
        set3.set_index('time')
        
        return [set1, set2, set3]
    
    def read_data (self, fileno, from_pickle= False):
        set_covariates= pd.read_csv(self.dataset_path + 'covariates_' + str(fileno) + '.csv')
        set_boot= pd.read_csv(self.dataset_path + 'boot_' + str(fileno) + '.csv')
        set_analytic= pd.read_csv(self.dataset_path + 'analytic_'  + str(fileno) + '.csv')

        if from_pickle== True:
            event_kl= self.read_pickle(self.dataset_path + "event_kl")
            event_sd= self.read_pickle(self.dataset_path + "event_sd")
        else:
            event_kl, event_sd, event_t = Event(self.dataset).make_events(set_analytic) 
        
        info_pack= {'KL': event_kl, 'SD': event_sd}

        return set_covariates, set_boot, info_pack

    def read_pickle (self, path: Path):
        file_handler = open(path, 'rb')
        obj = pickle.load(file_handler)
        file_handler.close()
        
        return obj
    