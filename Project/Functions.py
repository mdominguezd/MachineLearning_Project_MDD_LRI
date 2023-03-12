"""
Script with all of the functions used for the creation of the Machine Learning dashboard app.
"""
from typing import Union, Optional, Iterable, Tuple, Dict, List
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from datetime import datetime

## Calculation functions
import random
import numpy as np
from scipy.signal import convolve2d
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics.pairwise import pairwise_distances
import os

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
        
    train = pd.read_csv("https://gist.github.com/ionathan/3206e24244286dd25efd9e8bb39f079e/raw/ecde6eb8cdd47acb9dbfe2bfba7c241bfae19844/potato_NL_train.csv", sep=",")
    test  = pd.read_csv("https://gist.github.com/ionathan/3206e24244286dd25efd9e8bb39f079e/raw/ecde6eb8cdd47acb9dbfe2bfba7c241bfae19844/potato_NL_test.csv",  sep=",")
    
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

##############################################################################################
####################################CLASSIFICATION############################################
##############################################################################################

def open_raster(raster_path):
    ## Load image and make it readable as a NumPy matrix
    with rasterio.open(raster_path, 'r') as ds:
        return ds.read() 
    
def calculate_ndvi(red_band, nir_band):
    ## Calculates a new NDVI band from the red and the NIR band
    ndvi = (nir_band - red_band)/(nir_band + red_band)
    return ndvi

## Define the two functions needed to extract local statistics
def calculate_raster_local_means(raster, K):
    ## Function which loops over an entire raster to get local statistics
    ## for each individual band
    raster_local_means = []
    for _, raster_band in enumerate(raster):
        band_local_means = calculate_band_local_mean(raster_band, K)
        band_local_means = band_local_means[None, :]
        raster_local_means.append(band_local_means)
    raster_local_means = np.concatenate(raster_local_means, axis=0)
    
    return raster_local_means

def calculate_band_local_mean(band_matrix, K):
    ## Function that calculates the mean of a raster band within
    ## the square window size K
    ## Adjusted from https://www.delftstack.com/howto/python/moving-average-python/
    N = K ** 2
    padded_band_matrix = np.pad(band_matrix, 1, constant_values = 0)
    local_means = convolve2d(padded_band_matrix, np.ones([K,K]), 'valid') / N
    
    return local_means

def normalize_images(img1, img2):
    ## Function that normalizes the image by whitening it.
    ## We use the summary statistics of image 1 both for
    ## image 1 and 2 so that their distributions allign.
    img1_normalized = []
    img2_normalized = []
    for band_num, img1_band_matrix in enumerate(img1):
        # Calculate the summary statistics for the given band number for image 1
        img1_band_mean = np.mean(img1_band_matrix)
        img1_band_var = np.var(img1_band_matrix)
        img1_std = np.sqrt(img1_band_var)
        
        # Normalize image 1
        ''' FIL IN THE NORMALIZATION FUNCTION FOR THE BAND MATRIX OF IMAGE 1 HERE '''
        normalized_img1_band = (img1_band_matrix - img1_band_mean)/(img1_std)

        # Normalize image 2 using image 1 summary statistics
        img2_band_matrix = img2[band_num]
        ''' FIL IN THE NORMALIZATION FUNCTION FOR THE BAND MATRIX OF IMAGE 2 HERE '''
        normalized_img2_band = (img2_band_matrix - img1_band_mean)/(img1_std)

        # Append bands into a temporary list to later re-create the raster
        normalized_img1_band = normalized_img1_band[None, :]
        img1_normalized.append(normalized_img1_band)
        normalized_img2_band = normalized_img2_band[None, :]
        img2_normalized.append(normalized_img2_band)
    
    img1_normalized = np.concatenate(img1_normalized, )  
    img2_normalized = np.concatenate(img2_normalized)  
    
    return img1_normalized, img2_normalized

def read_data_cls():
    """
        Objective: Open the tif images as numpy arrays with all features as predictors and ground truth classes as variable to be predicted.
        
        Input:
            - None
        
        Output:
            - img1: First image that will be used for training
            - img2: Second image that will be used for test
            - img1_gt: Ground truth values for image 1
            - img2_gt: Ground truth values for image 2
        
    """
    # Correct order of bands
    corr_band_order = [2,1,0,3]
    # K parameter for lacal averages
    K = 3
    
    img1_rgb = open_raster("https://gist.github.com/ionathan/4369108d21efcba5743675add1c07def/raw/5d5dc834be9d2a50cff57d3f2bc545740cdb7f59/image1_rgb.tif")
    img1_nir = open_raster("https://gist.github.com/ionathan/4369108d21efcba5743675add1c07def/raw/5d5dc834be9d2a50cff57d3f2bc545740cdb7f59/image1_nir.tif")
    img1 = np.concatenate([img1_rgb, img1_nir], axis=0)
    img1 = np.array([img1[corr_band_order[0]],
                     img1[corr_band_order[1]],
                     img1[corr_band_order[2]],
                     img1[corr_band_order[3]],
                    ])
    # Calcuated NDVI value
    img1_ndvi = calculate_ndvi(img1[0], img1[3]+0.0000001)
    img1 = np.concatenate([img1, img1_ndvi[None, :, :]], axis=0)
    #add local means
    img1_local_means = calculate_raster_local_means(img1, K)
    img1 = np.concatenate([img1, img1_local_means], axis = 0)
    # Add extra features
    extra_features_img1 = ['https://gist.github.com/ionathan/4369108d21efcba5743675add1c07def/raw/5d5dc834be9d2a50cff57d3f2bc545740cdb7f59/im1_f'+str(f+1) + '.tif' for f in range(8)]
    img1_extra_features = [open_raster(f) for f in extra_features_img1]
    img1_extra_features = np.concatenate(img1_extra_features)
    img1 = np.concatenate([img1, img1_extra_features], axis=0)
    
    
    
    
    img2_rgb = open_raster("https://gist.github.com/ionathan/4369108d21efcba5743675add1c07def/raw/5d5dc834be9d2a50cff57d3f2bc545740cdb7f59/image2_rgb.tif")
    img2_nir = open_raster("https://gist.github.com/ionathan/4369108d21efcba5743675add1c07def/raw/5d5dc834be9d2a50cff57d3f2bc545740cdb7f59/image2_nir.tif")
    img2 = np.concatenate([img2_rgb, img2_nir], axis=0)
    img2 = np.array([img2[corr_band_order[0]],
                     img2[corr_band_order[1]],
                     img2[corr_band_order[2]],
                     img2[corr_band_order[3]],
                    ])
    # Calcuated NDVI values 
    img2_ndvi = calculate_ndvi(img2[0], img2[3]+0.0000001)
    img2 = np.concatenate([img2, img2_ndvi[None, :, :]], axis=0)
    # Add local means
    img2_local_means = calculate_raster_local_means(img2, K)
    img2 = np.concatenate([img2, img2_local_means], axis = 0)
    # Add extra features
    extra_features_img2 = ['https://gist.github.com/ionathan/4369108d21efcba5743675add1c07def/raw/5d5dc834be9d2a50cff57d3f2bc545740cdb7f59/im2_f'+str(f+1) + '.tif' for f in range(8)]
    img2_extra_features = [open_raster(f) for f in extra_features_img2]
    img2_extra_features = np.concatenate(img2_extra_features)
    img2 = np.concatenate([img2, img2_extra_features], axis=0)
    
    img1_gt = open_raster('https://gist.github.com/ionathan/4369108d21efcba5743675add1c07def/raw/5d5dc834be9d2a50cff57d3f2bc545740cdb7f59/image1_groundTruth.tif')
    img2_gt = open_raster('https://gist.github.com/ionathan/4369108d21efcba5743675add1c07def/raw/5d5dc834be9d2a50cff57d3f2bc545740cdb7f59/image2_groundTruth.tif')
    
    # Normalize images
    img1, img2 = normalize_images(img1, img2)
    
    return img1, img2, img1_gt, img2_gt


def data_clean_up(img1, img2, img1_gt, img2_gt):
    """
        Objective: Get the images, normalize them and export them as flattened numpy arrays.
        
        Input:
            - img1: First image that will be used for training
            - img2: Second image that will be used for test
            - img1_gt: Ground truth values for image 1
            - img2_gt: Ground truth values for image 2
        
        Output:
            - train_pixels: pixels with features used to train the model
            - test_pixels: pixels with features used to test the model
            - train_labels: pixels with labels used to train the model
            - test_labels: pixels with labels used to train the model
        
    """
    
    # Normalize images
    training_matrix, test_matrix = img1, img2
    
    # Labels
    training_matrix_labels = img1_gt
    test_matrix_labels = img2_gt[0,:,:]
    
    ## Flatten the splits so that they're 1-dimensional
    train_pixels = training_matrix.reshape(training_matrix.shape[0], training_matrix.shape[1] * training_matrix.shape[2])
    train_labels = training_matrix_labels[0].reshape(training_matrix.shape[1] * training_matrix.shape[2])

    test_pixels = test_matrix.reshape(test_matrix.shape[0], test_matrix.shape[1] * test_matrix.shape[2])
    test_labels = test_matrix_labels.reshape(test_matrix_labels.shape[0] * test_matrix_labels.shape[1])

    ## Rearrange the training array so that the dimension axis is last
    train_pixels = train_pixels.transpose()
    test_pixels = test_pixels.transpose()

    ## Remove background pixels from the arrays
    train_pixels = train_pixels[train_labels != 0, :]
    train_labels = train_labels[train_labels != 0]
    test_pixels = test_pixels[test_labels != 0, :]
    test_labels = test_labels[test_labels != 0]
    
    return train_pixels, test_pixels, train_labels, test_labels
    
def train_RFC(train_pixels, test_pixels, train_labels, test_labels, n_trees, min_samples):
    """
        Objective: Get the flattened numpy arrays and some random forest parameters
        
        Input:
            - train_pixels: pixels with features used to train the model
            - test_pixels: pixels with features used to test the model
            - train_labels: pixels with labels used to train the model
            - test_labels: pixels with labels used to train the model
            - n_terees: Nmber of trees for andom forest
            - min_samples: min_camples per leaf
        
        Output:
            - model: model fitted
            - time_taken: time takne to fit the model
            - predictions: predictions on test set
            - acc: accuracy on test set
    """
    ## Initialize the random forest classifier
    model = RandomForestClassifier(n_estimators=n_trees, 
                                   min_samples_split=min_samples,
                                   random_state=123)
    
    start = datetime.now()
    model.fit(train_pixels, train_labels)
    end = datetime.now()
    time_taken = float(f"{(end - start).seconds}.{round((end - start).microseconds, 2)}")
    
    ## Use the model to predict all of the validation pixels
    predictions = model.predict(test_pixels)
    
    #Calculate accuracy
    acc = accuracy_score(predictions, test_labels)
    
    return model, time_taken, predictions, acc

def apply_RFC(model, img2, img2_gt):
    """
        Objective: Apply the fitted model to the test set
        
        Input:
            - model: Model previously fitted
            - img2: Test image with the predictors
            - img2_gt: Test ground truth data
        
        Output:
            - results: Image classified
    """
    test_results_matrix = np.ones_like(img2_gt)

    ## Apply the model on every row of the input image.
    for xdim_index in range(img2.shape[1]):
        test_row = img2[:,xdim_index,:]
        test_row_preds = model.predict(test_row.transpose())
        test_results_matrix[:, xdim_index,:] = test_row_preds
        
    results = test_results_matrix[0]
                
    return results

def conf_matrix(true, predictions):
    """
        Objective: Calculate the confustion matrix
    """
    cm = confusion_matrix(true, predictions)
    
    return cm

def read_data_clu():
    """
        Objective: Function to read the data and normalize it 
    """
    ## Read the .csv files
    country_regions = pd.read_csv('https://gist.github.com/dmarcosg/d01e120a4e33d4c11de30091259ed51e/raw/c57e29837f4d4e900dc9a5c9227dcc770db639ae/country_codes.csv')
    oecd_cities_stats = pd.read_csv('https://gist.github.com/dmarcosg/d01e120a4e33d4c11de30091259ed51e/raw/c57e29837f4d4e900dc9a5c9227dcc770db639ae/oecd_cities_stats.csv')
    
    ## Extract the variables and their names
    data_cities = np.array(oecd_cities_stats.iloc[1:,1:],dtype=np.float32)
    number_of_variables = data_cities.shape[1]
    feature_names = list(oecd_cities_stats.columns.values[1:])
    
    ## Extract the country codes and region name of each city
    country_codes = []
    regions = []
    for name in oecd_cities_stats['Metropolitan areas'][1:]:
        country_codes.append(name[0:2])
        regions.append(country_regions[name[0:2]][1])

    regions = np.array(regions)
    country_codes = np.array(country_codes)
    
    data_cities = np.apply_along_axis(lambda x : (x - np.mean(x))/np.std(x), 0, data_cities)
    
    return data_cities, regions, feature_names

def my_kmeans(data, K=4, maxiter=10):
    """
        Objective: Function to perform K-means clustering in a np.array dataset.
        Inputs: 
            - data: np.arrray with the data to clusterize
            - K: number of clusters to create
            - maxiter: Iterations to perform
    """
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    ## TODO: Choose K data points as initial cluster centroids
    random_indeces = random.sample(range(len(data)),K)
    cluster_centroids = data[random_indeces]
    
    ## Loop over iterations
    for iter in range(maxiter):
        
        ## TODO: Compute distance between all data points and all centroids
        distances = pairwise_distances(data, cluster_centroids)
        
        ## TODO: Get the closest centroid to each data point
        cluster_assignments = np.argmin(distances, 1)
        
        ## TODO: Update the centroids
        for i in range(K):
            cluster_centroids[i,:] = np.mean(data[cluster_assignments == i], axis = 0)
            
    return cluster_assignments, cluster_centroids