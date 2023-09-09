import pandas as pd
import numpy as np
import statistics
import random
import re
import math
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import config as cfg

class DataETL:

    def __init__ (self, dataset):
        if dataset == "xjtu":
            self.total_bearings= cfg.N_BEARING_TOT_XJTU
            self.real_bearings= cfg.N_REAL_BEARING_XJTU
            self.total_signals= cfg.N_SIGNALS_XJTU
            self.boot_folder_size=  cfg.N_BOOT_FOLD_XJTU
            self.folder_size= cfg.N_BOOT_FOLD_UPSAMPLING
        elif dataset == "pronostia":
            self.total_bearings= cfg.N_BEARING_TOT_PRONOSTIA
            self.real_bearings= cfg.N_REAL_BEARING_PRONOSTIA
            self.total_signals= cfg.N_SIGNALS_PRONOSTIA
            self.boot_folder_size=  cfg.N_BOOT_FOLD_PRONOSTIA
            self.folder_size= cfg.N_BOOT_FOLD_UPSAMPLING

    def make_surv_data_sklS (self, covariates, set_boot, info_pack, bootstrap, type):
        row = pd.DataFrame()
        ref_value= {}
        data_cov= []
        type_foldering= []
        for i in range(0 , self.real_bearings ,1):
            data_cov.append(pd.DataFrame())
        for i in range(1, self.boot_folder_size * self.real_bearings, self.boot_folder_size):   
            type_foldering.append(range(i, (self.boot_folder_size * i) + 1, 1))

        for bear_num in range (1, self.total_bearings + 1, (bootstrap * 2) + 4):
            val= self.event_analyzer (bear_num, info_pack)
            ref_value.update({bear_num : val})
        
        moving_window= 0
        time_split= 20

        if type == 'bootstrap':
            for column in covariates:
                columnSeriesObj = covariates[column]
                columnSeriesObj= columnSeriesObj.dropna()

                bear_num = int(re.findall("\d?\d?\d", column)[0])
                temp_label_cov= ""
                
                if re.findall(r"mean\b", column):
                    temp_label_cov = "mean"
                elif re.findall(r"std\b", column):
                    temp_label_cov = "std"
                elif re.findall(r"skew\b", column):
                    temp_label_cov = "skew"
                elif re.findall(r"kurtosis\b", column):
                    temp_label_cov = "kurtosis"
                elif re.findall(r"entropy\b", column):
                    temp_label_cov = "entropy"
                elif re.findall(r"rms\b", column):
                    temp_label_cov = "rms"
                elif re.findall(r"max\b", column):
                    temp_label_cov = "max"
                elif re.findall(r"p2p\b", column):
                    temp_label_cov = "p2p"
                elif re.findall(r"crest\b", column):
                    temp_label_cov = "crest"
                elif re.findall(r"clearence\b", column):
                    temp_label_cov = "clearence"
                elif re.findall(r"shape\b", column):
                    temp_label_cov = "shape"
                elif re.findall(r"impulse\b", column):
                    temp_label_cov = "impulse"
                elif re.findall(r"FoH\b", column):
                    temp_label_cov = "FoH"
                elif re.findall(r"FiH\b", column):
                    temp_label_cov = "FiH"
                elif re.findall(r"FrH\b", column):
                    temp_label_cov = "FrH"
                elif re.findall(r"FrpH\b", column):
                    temp_label_cov = "FrpH"
                elif re.findall(r"FcaH\b", column):
                    temp_label_cov = "FcaH"
                elif re.findall(r"Fo\b", column):
                    temp_label_cov = "Fo"
                elif re.findall(r"Fi\b", column):
                    temp_label_cov = "Fi"
                elif re.findall(r"Fr\b", column):
                    temp_label_cov = "Fr"
                elif re.findall(r"Frp\b", column):
                    temp_label_cov = "Frp"
                elif re.findall(r"Fca\b", column):
                    temp_label_cov = "Fca"
                elif re.findall(r"noise\b", column):
                    temp_label_cov = "noise"
                elif re.findall(r"Event\b", column):
                    temp_label_cov = "Event"
                    columnSeriesObj = self.ev_manager (bear_num, bootstrap, self.total_bearings)
                elif re.findall(r"Survival_time\b", column):
                    temp_label_cov = "Survival_time"
                    columnSeriesObj = self.sur_time_manager(bear_num, set_boot, ref_value)
                
                label= temp_label_cov
                
                if label == "Event" or label == "Survival_time":
                    row [label]= pd.Series(columnSeriesObj).T   
                else:
                    row [label]= pd.Series(np.mean(columnSeriesObj.values)).T

                if label == "Survival_time":
                    for i, list in enumerate(type_foldering):
                        if bear_num in list:
                            data_cov[i] = pd.concat([data_cov[i], row], ignore_index= True)
        else:
            while moving_window < time_split:
                for column in covariates:
                    columnSeriesObj = covariates[column]
                    columnSeriesObj= columnSeriesObj.dropna()

                    bear_num = int(re.findall("\d?\d?\d", column)[0])
                    temp_label_cov= ""                    
                    
                    timepoints= int(self.sur_time_manager(bear_num, set_boot, ref_value))
                    time_window= int(timepoints / time_split)
                    if time_window == 0:
                        time_window = 2
                        low= time_window * moving_window
                        if moving_window < timepoints:
                            low= time_window * moving_window
                            high= time_window * (moving_window + 1)
                        else:
                            low= timepoints - 1
                            high= timepoints
                    else:                        
                        low= time_window * moving_window
                        high= time_window * (moving_window + 1)
                    
                    if re.findall(r"mean\b", column):
                        temp_label_cov = "mean"
                    elif re.findall(r"std\b", column):
                        temp_label_cov = "std"
                    elif re.findall(r"skew\b", column):
                        temp_label_cov = "skew"
                    elif re.findall(r"kurtosis\b", column):
                        temp_label_cov = "kurtosis"
                    elif re.findall(r"entropy\b", column):
                        temp_label_cov = "entropy"
                    elif re.findall(r"rms\b", column):
                        temp_label_cov = "rms"
                    elif re.findall(r"max\b", column):
                        temp_label_cov = "max"
                    elif re.findall(r"p2p\b", column):
                        temp_label_cov = "p2p"
                    elif re.findall(r"crest\b", column):
                        temp_label_cov = "crest"
                    elif re.findall(r"clearence\b", column):
                        temp_label_cov = "clearence"
                    elif re.findall(r"shape\b", column):
                        temp_label_cov = "shape"
                    elif re.findall(r"impulse\b", column):
                        temp_label_cov = "impulse"
                    elif re.findall(r"FoH\b", column):
                        temp_label_cov = "FoH"
                    elif re.findall(r"FiH\b", column):
                        temp_label_cov = "FiH"
                    elif re.findall(r"FrH\b", column):
                        temp_label_cov = "FrH"
                    elif re.findall(r"FrpH\b", column):
                        temp_label_cov = "FrpH"
                    elif re.findall(r"FcaH\b", column):
                        temp_label_cov = "FcaH"
                    elif re.findall(r"Fo\b", column):
                        temp_label_cov = "Fo"
                    elif re.findall(r"Fi\b", column):
                        temp_label_cov = "Fi"
                    elif re.findall(r"Fr\b", column):
                        temp_label_cov = "Fr"
                    elif re.findall(r"Frp\b", column):
                        temp_label_cov = "Frp"
                    elif re.findall(r"Fca\b", column):
                        temp_label_cov = "Fca"
                    elif re.findall(r"noise\b", column):
                        temp_label_cov = "noise"
                    elif re.findall(r"Event\b", column):
                        temp_label_cov = "Event"
                        columnSeriesObj = self.ev_manager (bear_num, bootstrap, self.total_bearings)
                    elif re.findall(r"Survival_time\b", column):
                        temp_label_cov = "Survival_time"
                        columnSeriesObj = self.sur_time_manager(bear_num, set_boot, ref_value)
                    
                    label= temp_label_cov
                    
                    if label == "Event" or label == "Survival_time":
                        if label == "Survival_time":
                            if type == "correlated":
                                if high < timepoints:
                                    proportional_value= columnSeriesObj - high
                                else:
                                    proportional_value= columnSeriesObj
                                row [label]= pd.Series(proportional_value).T
                            else:  
                                proportional_value= columnSeriesObj
                                row [label]= pd.Series(proportional_value).T  
                        else:
                            row [label]= pd.Series(columnSeriesObj).T   
                    else:
                        if type == "correlated":
                            if high < timepoints:
                                slice= columnSeriesObj.iloc[- (time_window * (moving_window + 1)): -1]
                            else:
                                slice= columnSeriesObj.iloc[low:-1]
                        else:                            
                            if high < timepoints:
                                slice= columnSeriesObj.iloc[low:high]
                            else:
                                slice= columnSeriesObj.iloc[low:-1]

                        row [label]= pd.Series(np.mean(slice.values)).T

                    if label == "Survival_time":
                        for i, list in enumerate(type_foldering):
                            if bear_num in list:
                                data_cov[i] = pd.concat([data_cov[i], row], ignore_index= True)
                
                moving_window+= 1

        return data_cov, ref_value
            
    def ev_manager (self, num, bootstrap, tot):
        checker= True

        #Only the two last bootstrapped bearings will be censored        
        censor_level= int((self.total_bearings / self.real_bearings) - 1)  
        for check in range(censor_level, tot + 1, (bootstrap * 2) + 4):
            if check == num or check == num - 1: 
                checker= False
                break
            else:
                checker= True
            
        if checker == False:
            return False
        else:
            return True

    def sur_time_manager (self, num, bootref, ref):
        for key, value in ref.items():
            if key == num or key + 1 == num:
                return value + random.randint(-2, 2)    
        
        bootstrap= len (bootref) - 1
        tot= ((bootstrap * 2) + 4) * self.real_bearings
        boot_pack_level=  int((self.total_bearings / self.real_bearings) + 1)
        boot_pack_max= int (self.total_bearings / self.real_bearings)
        num_ref= self.total_signals
       
        #Bootstrapping + addtitional randomizator
        i= 0
        for check in range(boot_pack_level, tot + (bootstrap * 2) + self.real_bearings, (bootstrap * 2) + 4):    
            if not num >= check:
                if num== num_ref:
                    return bootref.iat[0,i] + ref[check - boot_pack_max] + random.randint(-2, -1)               
                elif num== num_ref + 1:
                    return bootref.iat[0,i] + ref[check - boot_pack_max] + random.randint(1, 2)     
                elif num== num_ref + 2:
                    return bootref.iat[1,i] + ref[check - boot_pack_max] + random.randint(-2, -1)               
                elif num== num_ref + 3:
                    return bootref.iat[1,i] + ref[check - boot_pack_max] + random.randint(1, 2)   
                elif num== num_ref + 4:
                    return bootref.iat[2,i] + ref[check - boot_pack_max] + random.randint(-2, -1)              
                elif num== num_ref + 5:
                    return bootref.iat[2,i] + ref[check - boot_pack_max] + random.randint(1, 2)
                elif num== num_ref + 6:
                    return bootref.iat[3,i] + ref[check - boot_pack_max] + random.randint(-2, -1)              
                elif num== num_ref + 7:
                    return bootref.iat[3,i] + ref[check - boot_pack_max] + random.randint(1, 2)
                elif num== num_ref + 8:
                    return bootref.iat[4,i] + ref[check - boot_pack_max] + random.randint(-2, -1)              
                elif num== num_ref + 9:
                    return bootref.iat[4,i] + ref[check - boot_pack_max] + random.randint(1, 2)      #4
                elif num== num_ref + 10:
                    return bootref.iat[5,i] + ref[check - boot_pack_max] + random.randint(-2, -1)              
                elif num== num_ref + 11:
                    return bootref.iat[5,i] + ref[check - boot_pack_max] + random.randint(1, 2)      #5
                elif num== num_ref + 12:
                    return bootref.iat[6,i] + ref[check - boot_pack_max] + random.randint(-2, -1)              
                elif num== num_ref + 13:
                    return bootref.iat[6,i] + ref[check - boot_pack_max] + random.randint(1, 2)      #6
                elif num== num_ref + 14:
                    return bootref.iat[7,i] + ref[check - boot_pack_max] + random.randint(-2, -1)              
                elif num== num_ref + 15:
                    return bootref.iat[7,i] + ref[check - boot_pack_max] + random.randint(1, 2)      #7
                elif num== num_ref + 16:
                    return bootref.iat[8,i] + ref[check - boot_pack_max] + random.randint(-2, -1)              
                elif num== num_ref + 17:
                    return bootref.iat[8,i] + ref[check - boot_pack_max] + random.randint(1, 2)      #max bootstrap 8
                
            num_ref+= (bootstrap * 2) + 4
            i+= 1

        return -1

    def event_analyzer (self, bear_num, info_pack):
        lifetime_guarantee= 30
        data_kl= []
        data_sd= []

        for info in info_pack:
            for bear_info in info_pack [info][bear_num]:
                cross= bear_info [2] * 10  #10 as window
                tot_lenght= bear_info [3] * 10  #10 as window

                if info == "KL":
                    if cross > (tot_lenght)/100 * lifetime_guarantee:
                        data_kl.append(cross)
                elif info == "SD":
                    if cross > (tot_lenght)/100 * lifetime_guarantee:
                        data_sd.append(cross)

        if not data_kl:
            if data_sd:
                data_kl= data_sd
            else:
                data_kl= [tot_lenght]
                print("For bearing #{}, event considered at the end of the recording".format(bear_num))
        if not data_sd:
            if data_kl:
                data_sd= data_kl

        res = [max (data_kl), max (data_sd)]
        res= round (statistics.mean (res), 1)

        return res

    def format_main_data_Kfold (self, T1, train, test):
        ti_y_df= T1[0].iloc[train, -2:]
        cvi_y_df= T1[0].iloc[test, -2:]

        ti_y_df.rename(columns = {'Event':'event', 'Survival_time':'time'}, inplace = True)
        cvi_y_df.rename(columns = {'Event':'event', 'Survival_time':'time'}, inplace = True)

        ti_y_df.event = ti_y_df.event.replace({True: 1, False: 0})
        cvi_y_df.event = cvi_y_df.event.replace({True: 1, False: 0})

        ti_X = T1[0].iloc[train, :-2]
        ti_y = T1[1][train]
        cvi_X = T1[0].iloc[test, :-2]
        cvi_y = T1[1][test]

        ti_X.reset_index(inplace= True, drop=True)
        cvi_X.reset_index(inplace= True, drop=True)
        ti_y_df.reset_index(inplace= True, drop=True)
        cvi_y_df.reset_index(inplace= True, drop=True)

        # Collect splits
        ti = (ti_X, ti_y)
        cvi = (cvi_X, cvi_y)
        ti_NN = (ti_X, ti_y_df)
        cvi_NN = (cvi_X, cvi_y_df)

        return ti, cvi, ti_NN, cvi_NN

    def format_main_data (self, T1, T2):
        y_train_NN = T1[0].iloc[:, -2:]
        y_test_NN = T2[0].iloc[:, -2:]

        y_train_NN.rename(columns = {'Event':'event', 'Survival_time':'time'}, inplace = True)
        y_test_NN.rename(columns = {'Event':'event', 'Survival_time':'time'}, inplace = True)

        y_train_NN.event = y_train_NN.event.replace({True: 1, False: 0})
        y_test_NN.event = y_test_NN.event.replace({True: 1, False: 0})

        X_train = T1[0].iloc[:, :-2]
        y_train = T1[1]
        X_test = T2[0].iloc[:, :-2]
        y_test = T2[1]

        y_train_NN.reset_index(inplace= True, drop=True)
        y_test_NN.reset_index(inplace= True, drop=True)
        X_train.reset_index(inplace= True, drop=True)
        X_test.reset_index(inplace= True, drop=True)

        # Collect splits
        X_tr = (X_train, y_train)
        X_te = (X_test, y_test)
        y_tr_NN = (X_train, y_train_NN)
        y_te_NN = (X_test, y_test_NN)

        return X_tr, X_te, y_tr_NN, y_te_NN
    
    def centering_data (self, ti, cvi):
        ti_X = ti[0]
        ti_y = ti[1]
        cvi_X = cvi[0]
        cvi_y = cvi[1]
        features = list(ti_X.columns)

        # Apply scaling
        scaler = StandardScaler()

        scaler.fit(ti_X)
        ti_X = pd.DataFrame(scaler.transform(ti_X), columns=features)
        cvi_X = pd.DataFrame(scaler.transform(cvi_X), columns=features)

        ti_X.reset_index(inplace= True, drop=True)
        cvi_X.reset_index(inplace= True, drop=True)

        # Collect splits
        ti = (ti_X, ti_y)
        cvi = (cvi_X, cvi_y)

        return ti, cvi
    
    def centering_main_data (self, ti, cvi, ti_NN, cvi_NN):
        ti_X = ti[0]
        ti_y = ti[1]
        cvi_X = cvi[0]
        cvi_y = cvi[1]
        ti_X_NN = ti_NN[0]
        ti_y_NN = ti_NN[1]
        cvi_X_NN = cvi_NN[0]
        cvi_y_NN = cvi_NN[1]
        features = list(ti_X.columns)

        # Apply scaling
        scaler = StandardScaler()

        scaler.fit(ti_X)
        ti_X = pd.DataFrame(scaler.transform(ti_X), columns=features)
        cvi_X = pd.DataFrame(scaler.transform(cvi_X), columns=features)
        ti_X_NN = pd.DataFrame(scaler.transform(ti_X_NN), columns=features)
        cvi_X_NN = pd.DataFrame(scaler.transform(cvi_X_NN), columns=features)

        ti_X.reset_index(inplace= True, drop=True)
        cvi_X.reset_index(inplace= True, drop=True)
        ti_X_NN.reset_index(inplace= True, drop=True)
        cvi_X_NN.reset_index(inplace= True, drop=True)

        # Collect splits
        ti = (ti_X, ti_y)
        cvi = (cvi_X, cvi_y)
        ti_NN = (ti_X_NN, ti_y_NN)
        cvi_NN = (cvi_X_NN, cvi_y_NN)

        return ti, cvi, ti_NN, cvi_NN
    
    def format_centering_NN_data (self, T1NN, T2NN, ti_y_df, cvi_y_df, TvalNN):
        ti_X_NN = pd.concat([T1NN[0], ti_y_df], axis=1)
        cvi_X_NN = pd.concat([T2NN[0], cvi_y_df], axis=1)
        ti_X_val_NN= TvalNN[0]

        features = T1NN[0].columns
        cols_standardize = list(features)
        cols_leave = []

        scaler= StandardScaler()

        standardize = [([col], scaler) for col in cols_standardize]
        leave = [(col, None) for col in cols_leave]
        x_mapper = DataFrameMapper(standardize + leave)

        x_train_ti = x_mapper.fit_transform(ti_X_NN).astype('float32')
        x_train_cvi = x_mapper.fit_transform(cvi_X_NN).astype('float32')       
        x_val = x_mapper.transform(ti_X_val_NN).astype('float32')

        get_target = lambda df: (df['Survival_time'].values, df['Event'].values)
        y_ti_NN = get_target(ti_X_NN)
        y_val = get_target(ti_X_val_NN)

        durations_test, events_test = get_target(cvi_X_NN)       

        ti_NN = x_train_ti
        cvi_NN = x_train_cvi
        val_NN= x_val, y_val

        return ti_NN , y_ti_NN, cvi_NN, durations_test, events_test, val_NN

    def control_censored_data (self, X_test, y_test, percentage):
        censored_indexes = y_test.loc[:,"time"][y_test.loc[:,"event"]== 0].index

        #Drop censored data in percentage to avoid error in some models that is required 
        if len(censored_indexes) > 0:
            if np.floor(len(y_test)/100*percentage) == 0:
                num_censored= 1
            else:
                num_censored= int(np.floor(len(y_test)/100*percentage))
            censored_indexes = np.random.choice(censored_indexes, size= num_censored, replace=False)
        X_test.drop(censored_indexes, axis=0, inplace=True)
        X_test.reset_index(inplace= True, drop=True)
        y_test.drop(censored_indexes, axis=0, inplace=True)
        y_test.reset_index(inplace= True, drop=True)   

        return X_test, y_test
    
    def calculate_positions_percentages(self, dataframe, column_name, values):
        """
        Calculate the positions of values in a DataFrame column as percentages.

        Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the column.
        column_name (str): The name of the numerical column.
        values (list): List of numerical values for which you want to calculate the positions.

        Returns:
        list: List of positions of the values in percentages.
        """
        if not values:
            return []

        column = dataframe[column_name].tolist()
        column.sort()  # Sort the column in ascending order
        total_values = len(column)

        positions_percentages = []

        for value in values:
            if value < column[0]:
                position_percent = 0.0
            elif value > column[-1]:
                position_percent = 100.0
            else:
                # Find the index where the value would be inserted to maintain the order
                index = 0
                while index < total_values and column[index] < value:
                    index += 1
                
                # Calculate the position in percentage based on index and total values
                position_percent = (index / total_values) * 100
            
            positions_percentages.append(position_percent)
        
        return positions_percentages

    def find_values_by_percentages(self, dataframe, column_name, target_percentages):
        """
        Find values in a DataFrame column based on a list of target percentage values within the column.

        Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the column.
        column_name (str): The name of the numerical column.
        target_percentages (list): List of target percentage values.

        Returns:
        list: List of values from the column corresponding to the target percentage values.
        """
        if not target_percentages:
            return []
        
        column = dataframe[column_name].tolist()
        column.sort()  # Sort the column in ascending order
        total_values = len(column)
        
        values_by_percentages = []
        
        for target_percentage in target_percentages:
            target_position = int((target_percentage / 100) * total_values)
            values_by_percentages.append(column[target_position])
        
        return values_by_percentages

    def transform_column_with_cutpoints(dataset, cut_points, column_name):
        new_df = dataset.copy()
        for cut_point in cut_points:
            new_column_name = f"{column_name}_<=_{cut_point}"
            new_df[new_column_name] = dataset[column_name] >= cut_point
        
        return new_df


