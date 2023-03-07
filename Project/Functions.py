"""
Script with all of the functions used for the creation of the Machine Learning dashboard app.
"""
from typing import Union, Optional, Iterable, Tuple, Dict, List
import numpy as np
import pandas as pd

def read_data(split = True):
    """
        Objective: Split the data in predictors and predicted variable and read it as variables
        
        Input:
            - None
        
        Output:
            - X_train: predictors of training set
            - X_test: predictors of test set
            - y_train: predicted of training set
            - y_test: predicted of test set
        
    """
        
    train = pd.read_csv("yield_data/potato_NL_train.csv", sep=",")
    test  = pd.read_csv("yield_data/potato_NL_test.csv",  sep=",")
    
    if split:
        
        X_train = train.drop(columns=["IDREGION","FYEAR",'YIELD'])
        y_train = train['YIELD']

        X_test = test.drop(columns=["IDREGION","FYEAR",'YIELD'])
        y_test = test['YIELD']
    
        return X_train, X_test, y_train, y_test
    
    else:
        
        return train, test



def fit_regression(model : Union['LR', 'KNN', 'RFR'], X_train: Union[np.array, pd.DataFrame], X_test: Union[np.array, pd.DataFrame], y_train: Union[np.array, pd.DataFrame], y_test: Union[np.array, pd.DataFrame], K = 3, T = 100, D = None):
    """
        Objective: Fit the regession and predict the results on the test set.
        
        Input:
            - model: String (LR, KNN or RFR) specifying the regression to fit
            - X_train: predictors of training set
            - X_test: predictors of test set
            - y_train: predicted of training set
            - y_test: predicted of test set
            - K: parameter for knn regression (default = 3)
            - T: parameter for number of trees in RF (default = 100)
            - D: parameter for maximum depth of trees (default = None)
        
        Output:
            - fit_mod: fitted regression model
            - y_pred: predicted values for the test data
        
    """
    if model == 'LR':
        import statsmodels.api as sm
    
        fit_mod = sm.OLS(y_train, X_train).fit()
        
        y_pred = fit_mod.predict(X_test)
        
    elif model == 'KNN':
        from sklearn.neighbors import KNeighborsRegressor
        
        fit_mod = KNeighborsRegressor(n_neighbors=K).fit(X_train, y_train)
        
        y_pred = fit_mod.predict(X_test)
        
    elif model == 'RFR':
        from sklearn.ensemble import RandomForestRegressor
        
        fit_mod = RandomForestRegressor(n_estimators = T, max_depth = D).fit(X_train, y_train)
        
        y_pred = fit_mod.predict(X_test)
        
        
    else:
        raise ValueError("Model needs o be either 'LR', 'KNN' or 'RFR'")
    
    return fit_mod, y_pred




def get_metrics(y_true: Union[np.array, pd.DataFrame], y_pred: Union[np.array, pd.DataFrame]):
    """
        Objective: Get the accuracy metrics of a model and export them as a dictionary.
        
        Input:
            - y_true: True values of dependent variable (np.array or pd.DataFrame)
            - y_pred: Predicted values of dependent variable (np.array or pd.DataFrame)
        
        Output:
            - metrics: MAE, MSE, R^2
        
    """
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    MAE = mean_absolute_error(y_true, y_pred)
    MSE = mean_squared_error(y_true, y_pred)
    R2 = r2_score(y_true, y_pred)
    
    metrics = pd.DataFrame([MAE, MSE, R2], index = ['MAE', 'MSE', 'R^2']).T.apply(lambda df:round(df,2))
    metrics = metrics.to_dict('records')
    
    return metrics

def get_error_n_predicted_GeoJSON(nuts, model : Union['LR', 'KNN', 'RFR'], YEAR = 2012, K = 3, T = 100, D = None):
    """
        Objective: Get the predicted crop yields for the netherlands and return them as a JSON file
        
        Input:
            - nuts: DataFrame with geometries
            - model: Regression fitted to the data
            - Year: Year to be predicted
            - K: parameter for knn regression (default = 3)
            - T: parameter for number of trees in RF (default = 100)
            - D: parameter for maximum depth of trees (default = None)
        
        Output:
            - gdf_pred: geodataframe with predictions
            - gdf_error: geodataframe with errors
    """
    import geopandas as gpd
    
    X, X_, y, y_ = read_data()
    
    mod, y_p = fit_regression(model, X, X_, y, y_, K, T, D)
    
    df_error = pd.DataFrame([y_p - y_], index = ['Error']).T
    
    Tr, Te = read_data(split=False)
    
    # Whole dataframe
    DF = pd.concat([Tr,Te])
    
    DF['est_crop_yield'] = mod.predict(DF[DF.columns[2:-1]])
    
    DF = DF.groupby(['FYEAR', 'IDREGION']).mean()['est_crop_yield'].reset_index()
        
    DF_nuts2 = pd.merge(nuts, DF, left_on = "NUTS_ID", right_on = "IDREGION")
    
    DF_error = pd.concat([Te, df_error], axis = 1)
    DF_error = pd.merge(nuts, DF_error, left_on = "NUTS_ID", right_on = "IDREGION")
    
    gdf_pred = DF_nuts2[DF_nuts2['FYEAR']==YEAR]
    gdf_error = DF_error[DF_error['FYEAR']==YEAR]
    
    return gdf_pred, gdf_error
