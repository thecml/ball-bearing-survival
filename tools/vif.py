from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

class VIF:

    def __init__ (self, X, y, K):
        self.x = X
        self.y = y
        self.k = K
        self.selected_features= []

    def fit(self, X, y=None):
        # Create a DataFrame to store the VIF results
        data= X
        vif_data = pd.DataFrame()
        vif_data["Feature"] = data.columns

        vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(len(data.columns))]

        vif_data = vif_data.sort_values(by="VIF").reset_index(drop=True) 
        self.selected_features = vif_data.iloc[:self.k, :1]
        self.selected_features = self.selected_features["Feature"].values.tolist()
        return self

    def get_feature_names_out(self):
        return self.selected_features
