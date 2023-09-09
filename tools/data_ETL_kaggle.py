import pandas as pd
import numpy as np
import re

class DataETL_K:

    def __init__ (self):
        pass

    def make_covariates (self, dataframes):
        df_covariates= []
        i= 0
        
        for dataframe in dataframes:
            if i== 0:
                df_covariates.append(dataframe.drop(['B1_x_freq_band_1', 'B1_x_freq_band_2', 'B1_x_freq_band_3', 'B1_x_freq_band_4', 'B1_x_freq_band_5',
                                                    'B2_x_freq_band_1', 'B2_x_freq_band_2', 'B2_x_freq_band_3', 'B2_x_freq_band_4', 'B2_x_freq_band_5',
                                                    'B3_x_freq_band_1', 'B3_x_freq_band_2', 'B3_x_freq_band_3', 'B3_x_freq_band_4', 'B3_x_freq_band_5', 
                                                    'B4_x_freq_band_1', 'B4_x_freq_band_2', 'B4_x_freq_band_3', 'B4_x_freq_band_4', 'B4_x_freq_band_5',
                                                    'B1_y_freq_band_1', 'B1_y_freq_band_2', 'B1_y_freq_band_3', 'B1_y_freq_band_4', 'B1_y_freq_band_5',
                                                    'B2_y_freq_band_1', 'B2_y_freq_band_2', 'B2_y_freq_band_3', 'B2_y_freq_band_4', 'B2_y_freq_band_5',
                                                    'B3_y_freq_band_1', 'B3_y_freq_band_2', 'B3_y_freq_band_3', 'B3_y_freq_band_4', 'B3_y_freq_band_5', 
                                                    'B4_y_freq_band_1', 'B4_y_freq_band_2', 'B4_y_freq_band_3', 'B4_y_freq_band_4', 'B4_y_freq_band_5'], axis=1))
            elif i== 1:
                df_covariates.append(dataframe.drop(['B1_freq_band_1', 'B1_freq_band_2', 'B1_freq_band_3', 'B1_freq_band_4', 'B1_freq_band_5',
                                    'B2_freq_band_1', 'B2_freq_band_2', 'B2_freq_band_3', 'B2_freq_band_4', 'B2_freq_band_5',
                                    'B3_freq_band_1', 'B3_freq_band_2', 'B3_freq_band_3', 'B3_freq_band_4', 'B3_freq_band_5', 
                                    'B4_freq_band_1', 'B4_freq_band_2', 'B4_freq_band_3', 'B4_freq_band_4', 'B4_freq_band_5'], axis=1))
            elif i== 2:    
                df_covariates.append(dataframe.drop(['B1_freq_band_1', 'B1_freq_band_2', 'B1_freq_band_3', 'B1_freq_band_4', 'B1_freq_band_5',
                                    'B2_freq_band_1', 'B2_freq_band_2', 'B2_freq_band_3', 'B2_freq_band_4', 'B2_freq_band_5',
                                    'B3_freq_band_1', 'B3_freq_band_2', 'B3_freq_band_3', 'B3_freq_band_4', 'B3_freq_band_5', 
                                    'B4_freq_band_1', 'B4_freq_band_2', 'B4_freq_band_3', 'B4_freq_band_4', 'B4_freq_band_5'], axis=1))
                
            i += 1

        return df_covariates
    
    def make_surv_data_sklS (self, sets):
        row = pd.DataFrame()
        data_cov= pd.DataFrame()

        df_sa = pd.DataFrame([[False, 90.],[False, 92.], [True, 101.], [False, 102.], 
                    [True, 60.], [True, 75.], [True, 64.], [True, 62.],
                    [True, 57.], [True, 102.], [False, 100.], [False, 95.], 
                    [True, 96.], [False, 98.], [True, 91.], [True, 102.]], columns=['Event', 'Survival_time'])
        df_sa = df_sa[['Event', 'Survival_time']].to_numpy()
        df_sa = [(e1,e2) for e1,e2 in df_sa]
        data_sa= np.array(df_sa, dtype=[('Event', 'bool'), ('Survival_time', '<f8')])

        i= 1
        j= 1

        for set in sets:
            for column in set:

                if (column== "time"):
                    continue
                
                columnSeriesObj = set[column]
                temp_label_cov= ""

                if re.findall(r"mean\b", column):
                    temp_label_cov = re.findall(r"mean\b", column)[0]
                elif re.findall(r"std\b", column):
                    temp_label_cov = re.findall(r"std\b", column)[0]
                elif re.findall(r"skew\b", column):
                    temp_label_cov = re.findall(r"skew\b", column)[0]
                elif re.findall(r"kurtosis\b", column):
                    temp_label_cov = re.findall(r"kurtosis\b", column)[0]
                elif re.findall(r"entropy\b", column):
                    temp_label_cov = re.findall(r"entropy\b", column)[0]
                elif re.findall(r"rms\b", column):
                    temp_label_cov = re.findall(r"rms\b", column)[0]
                elif re.findall(r"max\b", column):
                    temp_label_cov = re.findall(r"max\b", column)[0]
                elif re.findall(r"p2p\b", column):
                    temp_label_cov = re.findall(r"p2p\b", column)[0]
                elif re.findall(r"crest\b", column):
                    temp_label_cov = re.findall(r"crest\b", column)[0]
                elif re.findall(r"clearence\b", column):
                    temp_label_cov = re.findall(r"clearence\b", column)[0]
                elif re.findall(r"shape\b", column):
                    temp_label_cov = re.findall(r"shape\b", column)[0]
                elif re.findall(r"impulse\b", column):
                    temp_label_cov = re.findall(r"impulse\b", column)[0]
                
                label= temp_label_cov
                row [label]= pd.Series(np.mean(columnSeriesObj.values)).T

                if i> 11:
                    i= 1
                    data_cov = pd.concat([data_cov, row], ignore_index= True)
                    j += 1
                else:
                    i += 1

            i= 1
            row = pd.DataFrame()

        return data_cov, data_sa

    def make_surv_data_pyS (self, sets):
        row = pd.DataFrame()
        data_sa = pd.DataFrame()
        data_cov= pd.DataFrame()

        event= [False, False, True, False, True, True, True, True, True, True, False, False, True, False, True, True]
        time_to_event= [90,92, 101, 102, 60, 75, 64, 62, 57, 102, 100, 95, 96, 98, 91, 102]

        data_sa ['Event']= pd.Series(event).T
        data_sa ['Survival_time']= pd.Series(time_to_event).T

        i= 1
        j= 1

        for set in sets:
            for column in set:

                if (column== "time"):
                    continue
                
                columnSeriesObj = set[column]
                temp_label_cov= ""

                if re.findall(r"mean\b", column):
                    temp_label_cov = re.findall(r"mean\b", column)[0]
                elif re.findall(r"std\b", column):
                    temp_label_cov = re.findall(r"std\b", column)[0]
                elif re.findall(r"skew\b", column):
                    temp_label_cov = re.findall(r"skew\b", column)[0]
                elif re.findall(r"kurtosis\b", column):
                    temp_label_cov = re.findall(r"kurtosis\b", column)[0]
                elif re.findall(r"entropy\b", column):
                    temp_label_cov = re.findall(r"entropy\b", column)[0]
                elif re.findall(r"rms\b", column):
                    temp_label_cov = re.findall(r"rms\b", column)[0]
                elif re.findall(r"max\b", column):
                    temp_label_cov = re.findall(r"max\b", column)[0]
                elif re.findall(r"p2p\b", column):
                    temp_label_cov = re.findall(r"p2p\b", column)[0]
                elif re.findall(r"crest\b", column):
                    temp_label_cov = re.findall(r"crest\b", column)[0]
                elif re.findall(r"clearence\b", column):
                    temp_label_cov = re.findall(r"clearence\b", column)[0]
                elif re.findall(r"shape\b", column):
                    temp_label_cov = re.findall(r"shape\b", column)[0]
                elif re.findall(r"impulse\b", column):
                    temp_label_cov = re.findall(r"impulse\b", column)[0]
                
                label= temp_label_cov
                
                row [label]= pd.Series(np.mean(columnSeriesObj.values)).T

                if i> 11:
                    i= 1
                    data_cov = pd.concat([data_cov, row], ignore_index= True)
                    j += 1
                else:
                    i += 1

            i= 1
            row = pd.DataFrame()

        return data_cov, data_sa  