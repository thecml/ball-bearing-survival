import numpy as np 
import pandas as pd
import scipy.stats as st
import statsmodels.stats.weightstats as stat 
from scipy.stats import entropy
import config as cfg

class Event:
        
    def __init__ (self, dataset):
        if dataset == "xjtu": 
            self.total_bearings = cfg.N_BEARING_TOT_XJTU
        elif dataset == "pronostia":
            self.total_bearings = cfg.N_BEARING_TOT_PRONOSTIA          
        self.window= 10
        self.percentage_error = 10
        self.break_in_percentage= 40
        self.t_threshold= - 1e-20

    def evaluator_KL (self, x, window):
        lenght= int((len(x) - x.iloc[:,0:1].isna().sum().values[0]) /window)
        len_col= len(x.columns)
        res = np.zeros((len_col,lenght - 1), dtype= float) #2156/36
        i = 0
        j = 0
        index_s = 0
        for x_label, __ in x.items():
            for index_s in range (window, len(x), window):
                index_e= index_s + window
                if index_e <= len(x) and not x [index_s : index_e].isnull().values.any():
                    temp_window1 = x [0 : window]
                else:
                    break
                temp_window2 = x [index_s : index_e]
                res [j][i] = entropy(temp_window1[x_label].values, temp_window2[x_label].values)
                i += 1
            i = 0
            j += 1

        return res

    def evaluator_Ttest (self, x, window):
        lenght= int((len(x) - x.iloc[:,0:1].isna().sum().values[0]) /window)
        len_col= len(x.columns)
        res = np.zeros((len_col,lenght - 1), dtype= float) #2156/36
        i = 0
        j = 0
        index_s = 0
        for x_label, __ in x.items():
            for index_s in range (window, len(x), window):
                index_e= index_s + window
                if index_e <= len(x) and not x [index_s : index_e].isnull().values.any():
                    temp_window1 = x [0 : window]
                else:
                    break
                temp_window2 = x [index_s : index_e]

                res [j][i] = stat.ttest_ind(temp_window1[x_label].values, temp_window2[x_label].values)[1]
                i += 1
            i = 0
            j += 1

        return res

    def evaluator_SD (self, x, window):
        lenght= int((len(x) - x.iloc[:,0:1].isna().sum().values[0]) /window)
        len_col= len(x.columns)
        res = np.zeros((len_col,lenght - 1), dtype= float) #2156/36
        i = 0
        j = 0
        index_s = 0
        for x_label, __ in x.items():
            for index_s in range (window, len(x), window):
                index_e= index_s + window
                if index_e > len(x) or x [index_s : index_e].isnull().values.any():
                    break
                temp_window = x [index_s : index_e]

                res [j][i] = np.std(temp_window[x_label].values)
                i += 1
            i = 0
            j += 1

        return res

    def evaluator_Chi (self, x, window):
        lenght= int((len(x) - x.iloc[:,0:1].isna().sum().values[0]) /window)
        len_col= len(x.columns)
        res = np.zeros((len_col,lenght - 1), dtype= float) #2156/36
        i = 0
        j = 0
        index_s = 0
        for x_label, __ in x.items():
            for index_s in range (window, len(x), window):
                index_e= index_s + window
                if index_e <= len(x) and not x [index_s : index_e].isnull().values.any():
                    temp_window1 = x [0 : window]
                else:
                    break
                temp_window2 = x [index_s : index_e]
                res [j][i] = st.chisquare(temp_window1[x_label].values, np.sum(temp_window1[x_label].values)/ np.sum(temp_window2[x_label].values) * temp_window2[x_label].values)[0]
                i += 1
            i = 0
            j += 1

        return res
    
    def evaluator_breakpoint (self, kl, sd, t, chi):
        i= 0
        j= 0
        w= 0
        m= 0
        q= 0
        bins = 5
        data= 3 + 1
        percentage_error= self.percentage_error
        break_in_percentage= self.break_in_percentage
        thresholds_kl = np.zeros((bins, data), dtype= float)
        thresholds_sd = np.zeros((bins, data), dtype= float)
        thresholds_t = np.zeros((bins, data), dtype= float)
        thresholds_chi = np.zeros((bins, data), dtype= float)

        for bin in kl:
            break_in= int(len(bin)* break_in_percentage /100)
            for step in bin:
                if i <= break_in:
                    if bin[i]> thresholds_kl [j][0] or i== 0:
                        thresholds_kl [j][0]= bin[i]
                        thresholds_kl [j][1]= bin[i] + (bin[i]/100 * percentage_error)
                        thresholds_kl [j][3]= len (bin)
                    w= i+1

                    if i> 0 and bin[i]>= thresholds_kl [j][0]:
                        thresholds_kl [j][2]= 0                    

                    for step_2 in bin[i+1:]:
                        if step_2 > thresholds_kl [j][1]:
                            m= bin[w]-bin[w-1]
                            q= bin[w-1]
                            thresholds_kl [j][2] =  (thresholds_kl [j][1]/m) - (q/m) + (w -1)
                            break
                        w += 1
                else:
                    break

                i+= 1
            i= 0
            j+= 1

        i= 0
        j= 0

        for bin in sd:
            break_in= int(len(bin)* break_in_percentage /100)
            for step in bin:
                if  i <= break_in:
                    if bin[i]> thresholds_sd [j][0] or i== 0:
                        thresholds_sd [j][0]= bin[i]
                        thresholds_sd [j][1]= bin[i] + (bin[i]/100 * percentage_error)
                        thresholds_sd [j][3]= len (bin)
                    w= i+1

                    if i> 0 and bin[i]>= thresholds_sd [j][0]:
                        thresholds_sd [j][2]= 0     

                    for step_2 in bin[i+1:]:
                        if step_2 > thresholds_sd [j][1]:
                            m= bin[w]-bin[w-1]
                            q= bin[w-1]
                            thresholds_sd [j][2] =  (thresholds_sd [j][1]/m) - (q/m) + (w -1)
                            break
                        w += 1
                else:
                    break

                i+= 1
            i= 0
            j+= 1

        i= 0
        j= 0
        count= 0
        t_threshold= self.t_threshold
        ground_level= 0
        max_count = 3

        for bin in t:
            break_in= int(len(bin)* break_in_percentage /100)
            thresholds_t [j][max_count]= len (bin)        
            for step in bin:
                if i == 0:
                    i+= 1
                    continue

                if count== data - 1:
                    break

                ground_level= min(bin) + (min(bin) /100 * percentage_error)

                x= list(range (0, len(bin), 1))
                y= bin
                dydx= np.diff(y)/np.diff(x)

                w= 1
                for val in dydx:
                    if val < t_threshold and bin[i] < ground_level and count< max_count and w> 1:
                        thresholds_t [j][count]= w
                        count+= 1
                    w += 1

                i+= 1
            count= 0    
            i= 0
            j+= 1

        i= 0
        j= 0
        count= 0
        c_threshold= self.t_threshold 
        ground_level= 0

        for bin in chi:
            break_in= int(len(bin)* break_in_percentage /100)
            thresholds_chi [j][max_count]= len (bin)
            for step in bin:
                if i == 0:
                    i+= 1
                    continue

                if count== data - 1:
                    break

                ground_level= min(bin) + (min(bin) /100 * percentage_error)

                x= list(range (0, len(bin), 1))
                y= bin
                dydx= np.diff(y)/np.diff(x)

                w= 1
                for val in dydx:
                    if val < c_threshold and bin[i] < ground_level and count< max_count and w> 1:
                        thresholds_chi [j][count]= w
                        count+= 1
                    w += 1

                i+= 1
            count= 0    
            i= 0
            j+= 1

        return thresholds_kl, thresholds_sd, thresholds_t, thresholds_chi
    

    def make_events (self, set_analytic):
        window= self.window
        eval_KL= []
        eval_t= []
        eval_SD= []
        eval_chi= []
        event_kl= []
        event_sd= []
        event_t= []
        event_chi= []         

        for b_num in range (1, self.total_bearings + 1, 1):
            set= set_analytic[["B{}_FoH".format(b_num), "B{}_FiH".format(b_num), "B{}_FrH".format(b_num), "B{}_FrpH".format(b_num), "B{}_FcaH".format(b_num)]]   
            eval_KL.append(self.evaluator_KL(set, window))
            eval_SD.append(self.evaluator_SD(set, window))
            eval_t.append(self.evaluator_Ttest(set, window))
            eval_chi.append(self.evaluator_Chi(set, window))

        i= 0
        for set_KL in eval_KL:
            temp_kl, temp_sd, temp_t, temp_chi= self.evaluator_breakpoint(set_KL, eval_SD[i], eval_t[i], eval_chi[i])
            event_kl.append(temp_kl)
            event_sd.append(temp_sd)
            event_t.append(temp_t)
            event_chi.append(temp_chi)
            i += 1

        return event_kl, event_sd, event_t