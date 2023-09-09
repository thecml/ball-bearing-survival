import numpy as np
import pandas as pd

class Survival:

    def __init__(self):
        pass    

    def predict_survival_function(self, model, X_test, times):
        # lower, upper = np.percentile(y_test[y_test.dtype.names[1]], [10, 90])
        # times = np.arange(np.ceil(lower + 1), np.floor(upper - 1), dtype=int)
        if model.__class__.__name__ == 'WeibullAFTFitter':
            surv_prob = model.predict_survival_function(X_test).T
            return surv_prob
        elif model.__class__.__name__ == 'DeepCoxPH' or model.__class__.__name__ == 'DeepSurvivalMachines':
            surv_prob = pd.DataFrame(model.predict_survival(X_test, t= list(times)), columns=times)
            return surv_prob          
        else:
            surv_prob = pd.DataFrame(np.row_stack([fn(times) for fn in model.predict_survival_function(X_test)]), columns=times)
            return surv_prob

    def predict_hazard_function(self, model, X_test, times):
        if model.__class__.__name__ == 'WeibullAFTFitter':
            surv_prob = model.predict_cumulative_hazard(X_test)
            return surv_prob
        elif model.__class__.__name__ == 'DeepCoxPH' or model.__class__.__name__ == 'DeepSurvivalMachines':
            risk_pred = model.predict_risk(X_test, t= times).flatten()
            return risk_pred            
        else:
            surv_prob = np.row_stack([fn(times) for fn in model.predict_cumulative_hazard_function(X_test)])
            return pd.DataFrame(surv_prob, columns=times)