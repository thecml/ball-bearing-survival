import pandas as pd
import numpy as np
import config as cfg
from abc import ABC, abstractmethod
from typing import List
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from mrmr import mrmr_regression
from tools.vif import VIF
import umap

class SelectAllFeatures ():
    def fit (self, X, y=None):
        X= X.loc[:, ~X.columns.isin(['Fca','Fi','Fo','Fr','Frp','FoH', 'FiH', 'FrH', 'FrpH', 'FcaH', 'noise'])]
        self.features = X.columns
        return self

    def get_feature_names_out (self):
        return self.features
    
class SelectPHFeatures ():
    def __init__ (self, X, y, data_info):
        self.x = X
        self.y = y
        self.data_info = data_info
        self.features = []

    def fit (self, X, y=None):
        if self.data_info[0] == "xjtu":
            if self.data_info[1] == "correlated":
                data_type = 'xjtu_corr'
            elif self.data_info[1] == "not_correlated":
                data_type =  'xjtu_not_corr'
            elif self.data_info[1] == "bootstrap":
                data_type =  'xjtu_boot'
        elif self.data_info[0] == "pronostia":
            if self.data_info[1] == "correlated":
                data_type = 'pronostia_corr'
            elif self.data_info[1] == "not_correlated":
                data_type =  'pronostia_not_corr'
            elif self.data_info[1] == "bootstrap":
                data_type =  'pronostia_boot'

        exclusion_list = cfg.PH_EXCLUSION[data_type]

        for feature in exclusion_list:
            X = X.drop(feature, axis=1)

        self.features = X.columns
        return self

    def get_feature_names_out (self):
        return self.features

def fit_and_score_features (X, y):
    n_features = X.shape[1]
    scores = np.empty(n_features)
    m = CoxPHSurvivalAnalysis(alpha= 0.4)
    for j in range(n_features):
        Xj = X[:, j:j+1]
        m.fit(Xj, y)
        scores[j] = m.score(Xj, y)
    return scores

class BaseFeatureSelector (ABC):
    """
    Base class for feature selectors.
    """
    def __init__ (self, X, y, estimator):
        """Initilizes inputs and targets variables."""
        self.X = X
        self.y = y
        self.estimator = estimator

    @abstractmethod
    def make_model (self):
        """
        """

    def get_features (self) -> List:
        ft_selector = self.make_model()
        if ft_selector.__class__.__name__ == "UMAP":
            self.fit(ft_selector, self.X)
            new_features = self.get_feature_names_out()
        else:
            ft_selector.fit(self.X, self.y)
            new_features = ft_selector.get_feature_names_out()
            print (new_features)
        return new_features

class NoneSelector (BaseFeatureSelector):
    def make_model (self):
        return SelectAllFeatures()
    
class PHSelector (BaseFeatureSelector):
    def make_model (self):
        return SelectPHFeatures(X= self.X, y= self.y, data_info= self.estimator)

class LowVar (BaseFeatureSelector):
    def make_model (self):
        return VarianceThreshold(threshold=0.001)

class SelectKBest4 (BaseFeatureSelector):
    def make_model (self):
        return SelectKBest(fit_and_score_features, k= 4)

class SelectKBest8 (BaseFeatureSelector):
    def make_model (self):
        return SelectKBest(fit_and_score_features, k= 8)

# class RFE4(BaseFeatureSelector):
#     def make_model(self):
#         return RFE_PI(self.estimator, n_features_to_select= 4, step=0.5)

# class RFE8(BaseFeatureSelector):
#      def make_model(self):
#          return RFE_PI(self.estimator, n_features_to_select= 8, step=0.5)

# class SFS4 (BaseFeatureSelector):
#     def make_model (self):
#         return SequentialFeatureSelector(self.estimator, n_features_to_select= 4,
#                                          scoring=fit_and_score_features,
#                                          direction="forward")

# class SFS8 (BaseFeatureSelector):
#     def make_model (self):
#         return SequentialFeatureSelector(self.estimator, n_features_to_select= 8,
#                                          scoring=fit_and_score_features,
#                                          direction="forward")

class VIF4 (BaseFeatureSelector):
    def make_model (self):
        return VIF(X= self.X, y= self.y, K= 4)
    
class VIF8 (BaseFeatureSelector):
    def make_model (self):
        return VIF(X= self.X, y= self.y, K=8)

class RegMRMR4 (BaseFeatureSelector):
    def make_model (self):
        return mrmr_regression(X= self.X, y= self.y, K= 4, show_progress= False)
    def get_features (self):
        return self.make_model()

class RegMRMR8 (BaseFeatureSelector):
     def make_model (self):
         return mrmr_regression(X= self.X, y= self.y, K= 8, show_progress= False)
     def get_features (self):
         return self.make_model()
    
class UMAP8 (BaseFeatureSelector):
    def make_model (self):
        self.components= 8
        return umap.UMAP(n_components= self.components,
                         n_neighbors= 6,
                         min_dist= 0.8,
                         metric= "manhattan")
    
    def fit (self, ft_selector, X, y= None):
        X = ft_selector.fit_transform(X)

        labels = []
        for element in range (1, self.components + 1 ,1):
            labels.append("UMAP_Feature_"+ str(element))

        self.features = pd.DataFrame(X, columns = labels)      
        return self

    def get_feature_names_out (self):
        return self.features
