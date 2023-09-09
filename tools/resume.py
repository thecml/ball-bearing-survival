import pandas as pd
import numpy as np
import dcor
import shap
import umap.plot
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from lifelines.utils import survival_table_from_events
from lifelines.statistics import proportional_hazard_test
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter

import config as cfg
import os
import re

class Resume:

    def __init__ (self, x, y, dataset):
        if dataset == "xjtu":
            self.result_path= cfg.RESULT_PATH_XJTU
            self.sample_path= cfg.SAMPLE_PATH_XJTU
        elif dataset == "pronostia":
            self.result_path= cfg.RESULT_PATH_PRONOSTIA
            self.sample_path= cfg.SAMPLE_PATH_PRONOSTIA
  
        self.hyper_results= cfg.HYPER_RESULTS   
        self.x= x
        self.y= y
        self.event_table= survival_table_from_events(x['Survival_time'].astype('int'),x['Event'])
        self.dpi= "figure"
        self.format= "png"
        self.test_size= 0.7

    def presentation (self, bearings, boot_no):
        x = self.x.iloc[:,:-2]
        x2= self.x.loc[:, ['p2p', 'max', 'clearence', 'mean', 'std', 'rms', 'crest', 'impulse', 'entropy','shape', 'kurtosis', 'skew', 'Survival_time']]
        considered_features = x.columns

        plt.rcParams.update({'font.size': 14})

        data_lab = []
        for num_bear in range(1, bearings + 1, 1):
            for num_boot in range(1, boot_no + 1, 1): 
                data_lab.append("Bearing {}".format(num_bear))
        
        labels= pd.DataFrame(data_lab, columns= ["Labels"])
        
        #Plot censoring distribution
        n_censored = self.y.shape[0] - self.y["Event"].sum()
        print ("There are {} samples censored".format(n_censored))
        print("%.1f%% of records are censored" % (n_censored / self.y.shape[0] * 100))
        plt.figure(figsize=(9, 6))
        plt.xlabel("Time (10 min)")
        plt.ylabel("Number of occurrences")
        val, bins, patches = plt.hist((self.y["Survival_time"][self.y["Event"]],
                                       self.y["Survival_time"][~self.y["Event"]]),
                                       bins=50, stacked=True)
        _ = plt.legend(patches, ["Time of Death", "Time of Censoring"])
        plt.savefig(self.result_path + 'censoring_data.png', dpi= self.dpi, format= self.format, bbox_inches='tight')
        plt.close()

        #Plot UMAP
        mapper = umap.UMAP(n_neighbors= 6,
                         min_dist= 0.8,
                         metric="manhattan").fit(x) 
        umap.plot.connectivity(mapper, show_points=True)
        plt.savefig(self.result_path + 'UMAP_conn.png', dpi= self.dpi, format= self.format, bbox_inches='tight')
        plt.close()
        umap.plot.points(mapper, labels= labels["Labels"])
        plt.savefig(self.result_path + 'UMAP_class.png', dpi= self.dpi, format= self.format, bbox_inches='tight')
        plt.close()

        #Plot linear multicollinearity
        plt.figure(figsize=(10,7))
        mask = np.triu(np.ones_like(x.corr(), dtype=bool))
        sns.heatmap(x.corr(), annot=True, mask=mask, vmin=-1, vmax=1)
        # plt.show()
        plt.savefig(self.result_path + 'lin_multicorr_x.png', dpi= self.dpi, format= self.format, bbox_inches='tight')
        plt.close()

        #Compute vif 
        self.compute_vif(considered_features).sort_values('VIF', ascending=False)

        #Plot linear multicollinearity with output
        plt.figure(figsize=(10,7))
        mask = np.triu(np.ones_like(x2.corr(), dtype=bool))
        sns.heatmap(x2.corr(), annot=True, mask=mask, vmin=-1, vmax=1)
        # plt.show()
        plt.savefig(self.result_path + 'lin_multicorr_xy.png', dpi= self.dpi, format= self.format, bbox_inches='tight')
        plt.close()

        #Plot non-linear multicollinearity
        correlation_matrix = np.zeros((len(x.columns), len(x.columns)))
        for i in range(len(x.columns)):
            for j in range(i+1, len(x.columns)):
                feature1 = x.iloc[:, i].values.reshape(-1, 1)
                feature2 = x.iloc[:, j].values.reshape(-1, 1)
                NLcor = dcor.distance_correlation(feature1, feature2)
                correlation_matrix[i, j] = NLcor
                correlation_matrix[j, i] = NLcor

        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        plt.figure(figsize=(10,7))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', mask=mask, vmin=-1, vmax=1,
                    xticklabels=x.columns, yticklabels=x.columns)
        # plt.show()
        plt.savefig(self.result_path + 'nonlin_multicorr_x.png', dpi= self.dpi, format= self.format, bbox_inches='tight')
        plt.close()

        #Plot non-linear multicollinearity with output
        correlation_matrix = np.zeros((len(x2.columns), len(x2.columns)))
        for i in range(len(x2.columns)):
            for j in range(i+1, len(x2.columns)):
                feature1 = x2.iloc[:, i].values.reshape(-1, 1)
                feature2 = x2.iloc[:, j].values.reshape(-1, 1)
                NLcor = dcor.distance_correlation(feature1, feature2)
                correlation_matrix[i, j] = NLcor
                correlation_matrix[j, i] = NLcor

        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        plt.figure(figsize=(10,7))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', mask=mask, vmin=-1, vmax=1,
                    xticklabels= x2.columns, yticklabels= x2.columns)
        # plt.show()
        plt.savefig(self.result_path + 'nonlin_multicorr_xy.png', dpi= self.dpi, format= self.format, bbox_inches='tight')
        plt.close()

        #Check PH assumption is made outside this routine once
        # cph = CoxPHFitter()
        # cph.fit(self.x, duration_col= "Survival_time", event_col= "Event")
        # cph.check_assumptions(self.x, p_value_threshold=0.05, show_plots=True)
        # results = proportional_hazard_test(cph, self.x, time_transform='rank')
        # results.print_summary(decimals=3, model="untransformed variables")

        km_sc= KaplanMeierFitter()
        km_sc.fit(durations= self.x["Survival_time"], event_observed= self.x["Survival_time"])
        km_sc.predict(11)
        km_sc.plot(figsize=(20, 20), linewidth=2)
        plt.xlabel("Time (10 min)")
        plt.ylabel("Survival probability")
        plt.title("Kaplan Meier")
        plt.savefig(self.result_path + 'KM_line.png', dpi= self.dpi, format= self.format, bbox_inches='tight')
        plt.close()

    def plot_simple_sl (self, y_test, surv_probs, model):
        plt.rcParams.update({'font.size': 14})
        surv_label= []
        for i in range (0, len (y_test) +1):
            surv_label.append('Bearing ' + str(i) + ' test')

        surv_probs.T.plot(figsize=(20, 20), linewidth= 2)            
        plt.xlabel("Time (10 min)")
        plt.ylabel("Survival probability")
        plt.grid()
        plt.legend(surv_label)
        plt.plot ()
        # plt.title("{}".format(model))
        # plt.savefig(self.result_path + 'sl_{}.png'.format(model) , dpi= self.dpi, format= self.format, bbox_inches='tight')
        # plt.close()

    def plot_aggregate_sl (self, km_sc, survival_probs):
            plt.rcParams.update({'font.size': 12})
            plt.figure(dpi=80)

            surv_label= []
            surv_label.append('Weibull')
            surv_label.append('CoxPH')
            surv_label.append('RSF')
            surv_label.append('CoxBoost')
            surv_label.append('DeepSurv')
            surv_label.append('DSM')
            surv_label.append('KM + 95% CI')

            for survival_prob in survival_probs:
                probs= np.mean(survival_prob, axis= 0)
                probs.T.plot(linewidth=1.5)

            km_sc.plot(linewidth=2, alpha=0.4)

            plt.xlabel("Time (10 min)")
            plt.ylabel("Survival probability S(t)")
            plt.legend(surv_label)
            plt.grid()

    def plot_sl_ci (self, y_test, surv_probs, model):
        plt.rcParams.update({'font.size': 14})
        event_table= self.event_table
        if model.__class__.__name__ == 'DeepCoxPH':
            ref_prob_high = surv_probs.index[-1]
            ref_prob_low = surv_probs.index[0]
        else:
            ref_prob_high = surv_probs.T.index[-1]
            ref_prob_low = surv_probs.T.index[0]
        filter_idx_high= self.find_largest_below_threshold(event_table.index, ref_prob_high)
        filter_idx_low= self.find_smallest_over_threshold(event_table.index, ref_prob_low)
        event_table_idx= event_table.index
        new_event_table = [x for x in event_table_idx if x <= filter_idx_high and x >= filter_idx_low]

        minn= 0
        maxn= 1

        result= []
        survival_functions= []

        for i in range (0, len (y_test)):
            if model.__class__.__name__ == 'DeepCoxPH':
                survival_function = surv_probs[i][new_event_table]
            else:
                survival_function = surv_probs.T[i][new_event_table]                
            # Calculate the Greenwood formula
            n_events = event_table.iloc[:, 0]
            variance_estimate = np.cumsum(event_table['observed'] / (event_table['at_risk'] * (event_table['at_risk'] - event_table['observed'])))

            # Calculate the confidence intervals using the Greenwood formula
            z = 1.96  # Z-value for 95% confidence interval
            lower_bound = np.clip(survival_function * np.exp(-z * np.sqrt(variance_estimate / (survival_function ** 2))), minn, maxn)
            upper_bound = np.clip(survival_function * np.exp(z * np.sqrt(variance_estimate / (survival_function ** 2))), minn, maxn)
            result_temp = pd.DataFrame({'Survival Function': survival_function,
                                'Lower Bound': lower_bound, 'Upper Bound': upper_bound})
            result_temp = result_temp.dropna()
            survival_functions.append(survival_function)    
            result.append(result_temp)

        surv_label = []
        for i in range (1, len (y_test) + 2):
            surv_label.append('Bearing ' + str(i) + ' test')
            surv_label.append('Confidence Interval 5-95% ' + 'B' + str(i))

        i= 0
        for sf in survival_functions:
            sf.plot(figsize=(20, 20), linewidth= 2)
            plt.fill_between(result[i].index, (result[i]["Lower Bound"].values), (result[i]["Upper Bound"].values), alpha=.1)
            i += 1
        plt.xlabel("Time (10 min)")
        plt.ylabel("Survival probability")
        plt.legend(surv_label)
        plt.grid()
        plt.title("{}".format(model))
        plt.savefig(self.result_path + 'sl_CI_{}.png'.format(model) , dpi= self.dpi, format= self.format, bbox_inches='tight')
        plt.close()

    def plot_shap (self, explainer, shap_values, X_test, model):

        if model != "Neural Network":
            shap.plots.waterfall(shap_values[0], show=False)
            plt.savefig(self.result_path + 'waterfall_{}.png'.format(model) , dpi= self.dpi, format= self.format, bbox_inches='tight')
            plt.close()
            shap.plots.beeswarm(shap_values, show=False)
            plt.savefig(self.result_path + 'beeswarm_{}.png'.format(model) , dpi= self.dpi, format= self.format, bbox_inches='tight')
            plt.close()           
        else:
            idx = 3
            exp = shap.Explanation(shap_values.values, shap_values.base_values[0], shap_values.data)
            print (exp[idx])
            shap.plots.waterfall(exp[idx])
            plt.savefig(self.result_path + 'waterfall_{}.png'.format(model) , dpi= self.dpi, format= self.format, bbox_inches='tight')
            plt.close()                  
            shap.plots.beeswarm(shap_values)
            plt.savefig(self.result_path + 'beeswarm_{}.png'.format(model) , dpi= self.dpi, format= self.format, bbox_inches='tight')
            plt.close() 

    def plot_performance (self, last, df_CI, df_B, model_name= None, CI_score= None, brier_score= None):
        if last == True:
            _, ax = plt.subplots(figsize=(11, 6))
            sns.boxplot(x= "Model", y="CI score", data=df_CI, ax=ax)
            sns.set_style("whitegrid")
            _, xtext = plt.xticks()
            for t in xtext:
                t.set_rotation("vertical")
            plt.savefig(self.result_path + 'CI_results.png', dpi= self.dpi, format= self.format, bbox_inches='tight')
            plt.close()        

            _, ax = plt.subplots(figsize=(11, 6))
            sns.boxplot(x= "Model", y="Brier score", data=df_B, ax=ax)
            sns.set_style("whitegrid")
            _, xtext = plt.xticks()
            for t in xtext:
                t.set_rotation("vertical")
            plt.savefig(self.result_path + 'brier_results.png', dpi= self.dpi, format= self.format, bbox_inches='tight')
            plt.close()   

            return df_CI, df_B
        else:
            temp_CI= {"Model": [], "CI score": []}
            temp_CI["Model"].append(model_name)
            temp_CI["CI score"].append(CI_score)
            temp_CI = pd.DataFrame.from_dict(temp_CI)
            df_CI = pd.concat([df_CI, temp_CI], ignore_index= True)

            if brier_score!= None:
                temp_B= {"Model": [], "Brier score": []}
                temp_B["Model"].append(model_name)
                temp_B["Brier score"].append(brier_score)    
                temp_B = pd.DataFrame.from_dict(temp_B)
                df_B = pd.concat([df_B, temp_B], ignore_index= True)

            return df_CI, df_B

    def table_result_hyper(self):

        itr = os.walk(self.hyper_results)
        cph_results = str()
        dl_results = str()
        rsf_results = str()
        cb_results = str()
        aft_results = str()
        next(itr)
        pd.set_option('use_inf_as_na',True)

        for next_root, next_dirs, next_files in itr: 
            itr_final = os.walk(next_root)
            next(itr_final)
            
            for final_root, final_dirs, final_files in itr_final: 
                
                for filename in os.listdir(final_root):
                    if re.findall(r"\bCoxPH", filename):
                        cph_results = pd.read_csv(os.path.join(final_root, filename))
                    elif re.findall(r"\bDeepSurv", filename):
                        dl_results = pd.read_csv(os.path.join(final_root, filename))
                    elif re.findall(r"\bRSF", filename):
                        rsf_results = pd.read_csv(os.path.join(final_root, filename))
                    elif re.findall(r"\bCoxBoost", filename):
                        cb_results = pd.read_csv(os.path.join(final_root, filename))
                    elif re.findall(r"\bWeibullAFT", filename):
                        aft_results = pd.read_csv(os.path.join(final_root, filename))

                cv_results = pd.concat([cph_results, rsf_results, cb_results, dl_results, aft_results], axis=0)
                cv_results=cv_results.dropna().reset_index(drop=True)


                col_order = ['(1) None', '(2) VIF4', '(3) SelectKBest4', '(4) SelectKBest8', '(5) RegMRMR4', '(6) RegMRMR8']
                row_order = ['(1) CoxPH', '(2) RSF', '(3) CoxBoost', '(4) DeepSurv', '(5) WeibullAFT'] 

                # Group results for heatmaps
                cv_grp_results = cv_results.groupby(['ModelName', 'FtSelectorName'])[['CIndex', 'BrierScore']] \
                                .mean().round(4).reset_index()
                
                c_index_res = cv_grp_results.pivot(index='ModelName', columns=['FtSelectorName'], values=['CIndex']) \
                                            .rename_axis(None, axis=0).set_axis(range(0, len(col_order)), axis=1) \
                                            .set_axis(col_order, axis=1).reindex(row_order)
                
                brier_score_res = cv_grp_results.pivot(index='ModelName', columns=['FtSelectorName'], values=['BrierScore']) \
                                                .rename_axis(None, axis=0).set_axis(range(0, len(col_order)), axis=1) \
                                                .set_axis(col_order, axis=1).reindex(row_order)
        #        brier_score_res = brier_score_res.apply(lambda x: 100 - (x * 100)) # for better readability

                c_index_res.xs('(5) WeibullAFT')['(4) SelectKBest8'] = np.nan
                c_index_res.xs('(5) WeibullAFT')['(5) RegMRMR4'] = np.nan
                c_index_res.xs('(5) WeibullAFT')['(6) RegMRMR8'] = np.nan

                brier_score_res.xs('(5) WeibullAFT')['(4) SelectKBest8'] = np.nan
                brier_score_res.xs('(5) WeibullAFT')['(5) RegMRMR4'] = np.nan
                brier_score_res.xs('(5) WeibullAFT')['(6) RegMRMR8'] = np.nan

                data = cv_grp_results.loc[cv_grp_results['ModelName'] == '(1) CoxPH']['CIndex']

                # Plot heatmap of c-index
                df = pd.DataFrame(c_index_res)
                annot_df = df.applymap(lambda f: f'{f:.3g}')
                fig, ax = plt.subplots(figsize=(25, 7), squeeze=False)
                sns.heatmap(np.where(df.isna(), 0, np.nan), ax=ax[0, 0], cbar=False,
                            annot=np.full_like(df, "NA", dtype=object), fmt="",
                            annot_kws={"size": 14, "va": "center_baseline", "color": "black"},
                            cmap=sns.diverging_palette(20, 220, n=200), linewidth=0)
                sns.heatmap(df, ax=ax[0, 0], cbar=True, annot=annot_df,
                            fmt="", annot_kws={"size": 14, "va": "center_baseline"},
                            cmap=sns.diverging_palette(20, 220, n=200),#vmin=0.5, vmax=1,
                            linewidth=2, linecolor="black", xticklabels=True, yticklabels=True)
                ax[0,0].set_ylabel('Machine Learning Model', fontsize=14)
                ax[0,0].set_xlabel('Feature Selection Method', fontsize=14)
                ax[0,0].xaxis.set_ticks_position('top')
                ax[0,0].xaxis.set_label_position('top')
                ax[0,0].tick_params(axis='both', which='major', labelsize=14)
                plt.xticks(rotation=45)
                plt.savefig(final_root + "cindex_table.png")

                # Plot heatmap of brier score
                df = pd.DataFrame(brier_score_res)
                annot_df = df.applymap(lambda f: f'{f:.3g}')
                fig, ax = plt.subplots(figsize=(25, 8), squeeze=False)
                sns.heatmap(np.where(df.isna(), 0, np.nan), ax=ax[0, 0], cbar=False,
                            annot=np.full_like(df, "NA", dtype=object), fmt="",
                            annot_kws={"size": 14, "va": "center_baseline", "color": "black"},
                            cmap=sns.diverging_palette(20, 220, n=200), linewidth=0)
                sns.heatmap(df, ax=ax[0, 0], cbar=True, annot=annot_df, 
                            fmt="", annot_kws={"size": 14, "va": "center_baseline"},
                            cmap=sns.diverging_palette(20, 220, n=200),# vmin=3, vmax=5.9,
                            linewidth=2, linecolor="black", xticklabels=True, yticklabels=True)
                ax[0,0].set_ylabel('Machine Learning Model', fontsize=14)
                ax[0,0].set_xlabel('Feature Selection Method', fontsize=14)
                ax[0,0].xaxis.set_ticks_position('top')
                ax[0,0].xaxis.set_label_position('top')
                ax[0,0].tick_params(axis='both', which='major', labelsize=14)
                plt.xticks(rotation=45)
                plt.savefig(final_root + "brier_table.png")

                # Make table with ci results
                c_index_mean = cv_results.groupby(['ModelName', 'FtSelectorName'])[['CIndex']].mean().round(2)
                c_index_std = cv_results.groupby(['ModelName', 'FtSelectorName'])[['CIndex']].std().round(2)
                col_order = cv_results['FtSelectorName'].unique()
                row_order = ['(1) CoxPH', '(2) RSF', '(3) CoxBoost', '(4) DeepSurv', '(5) WeibullAFT']
                results_merged = pd.merge(c_index_mean, c_index_std, left_index=True,
                                        right_index=True, suffixes=('Mean', 'Std')).reset_index()
                results_merged = results_merged.fillna("NA")
                results_merged['CIndex'] = results_merged['CIndexMean'].astype(str) + " ($\pm$"+ results_merged["CIndexStd"].astype(str) + ")"
                table = results_merged.pivot(index='ModelName', columns=['FtSelectorName'], values=['CIndex']) \
                            .rename_axis(None, axis=0).set_axis(range(0, len(col_order)), axis=1) \
                            .set_axis(col_order, axis=1).reindex(row_order)
                file = open(final_root + 'Latex_CI.txt', 'w')
                file.write(table.style.to_latex())
                file.close()

                # Make table with brier results
                bri_mean = cv_results.groupby(['ModelName', 'FtSelectorName'])[['BrierScore']].mean().round(2)
                bri_std = cv_results.groupby(['ModelName', 'FtSelectorName'])[['BrierScore']].std().round(2)
                col_order = cv_results['FtSelectorName'].unique()
                row_order = ['(1) CoxPH', '(2) RSF', '(3) CoxBoost', '(4) DeepSurv', '(5) WeibullAFT']
                results_merged = pd.merge(bri_mean, bri_std, left_index=True,
                                        right_index=True, suffixes=('Mean', 'Std')).reset_index()
                results_merged = results_merged.fillna("NA")
                results_merged['BrierScore'] = results_merged['BrierScoreMean'].astype(str) + " ($\pm$"+ results_merged["BrierScoreStd"].astype(str) + ")"
                table = results_merged.pivot(index='ModelName', columns=['FtSelectorName'], values=['BrierScore']) \
                            .rename_axis(None, axis=0).set_axis(range(0, len(col_order)), axis=1) \
                            .set_axis(col_order, axis=1).reindex(row_order)
                file = open(final_root + 'Latex_bri.txt', 'w')
                file.write(table.style.to_latex())
                file.close()
            
            pd.set_option('use_inf_as_na',True)       

    def compute_vif (self, considered_features):
        x = self.x[considered_features]
        
        #The calculation of variance inflation requires a constant
        x['intercept'] = 1
        
        vif = pd.DataFrame()
        vif["Variable"] = x.columns
        vif["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
        vif = vif[vif['Variable']!='intercept']

        return vif
    
    def find_largest_below_threshold (self, array, threshold):
        largest_number = 0
        for num in array:
            if num <= threshold and num >= largest_number:
                largest_number = num

        return largest_number
    
    def find_smallest_over_threshold (self, array, threshold):
        smallest_number = float('inf')
        for num in array:
            if num >= threshold and num <= smallest_number:
                smallest_number = num

        return smallest_number

class _TFColor(object):
    """Enum of colors used in TF docs."""
    red = '#F15854'
    blue = '#5DA5DA'
    orange = '#FAA43A'
    green = '#60BD68'
    pink = '#F17CB0'
    brown = '#B2912F'
    purple = '#B276B2'
    yellow = '#DECF3F'
    gray = '#4D4D4D'
    def __getitem__(self, i):
        return [
            self.red,
            self.orange,
            self.green,
            self.blue,
            self.pink,
            self.brown,
            self.purple,
            self.yellow,
            self.gray,
        ][i % 9]