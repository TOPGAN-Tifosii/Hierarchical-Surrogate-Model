#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.metrics import r2_score
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import pickle
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)


# In[ ]:


# Code for training the subwatershed surrogate model
# First, the production method of subDD_sub_num.csv needs to be explained.

# Please enter the number of subbasins at “XX” below
NOS = XX

for SUB in range(1, NOS + 1):
    sub_num = "{:02}".format(SUB)
    # Read the data of a single subwatershed named "subDD_xx.csv"
    dataset = pd.read_csv('subDD_' + sub_num + '.csv')
    # Use the get_dummies function to convert the GW column into one - hot encode
    gw_encoded = pd.get_dummies(dataset['GW'], prefix='GW')

    # Merge the one - hot encoded columns with the original data
    dataset = pd.concat([dataset, gw_encoded], axis = 1)

    # Delete the original "GW" column
    dataset.drop('GW', axis = 1, inplace = True)
    # Split the data into features and labels
    features = dataset.drop(['SUB', 'YEAR', 'WYLDmm', 'SYLDt_ha', 'subTN', 'BMPS', 'FLOW_OUTcms', 'SED_OUTtons', 'TOT_Nkg',
                             'ORGNkg_ha', 'NSURQkg_ha', 'LAT_Q_NO3kg_ha', 'GWNO3kg_ha', 'MON'], axis = 1)
    labels = dataset[['WYLDmm', 'SYLDt_ha', 'subTN']]

    # Separate the data to be standardized from the BMPs data
    scaler_features = features[['PCP', 'RH', 'MAXTM', 'MINTM', 'WIND']]
    BMPs_features = features[['FSW', 'FSC', 'FRT_off', 'GW_0', 'GW_1']]
    # Reset the index
    BMPs_features = BMPs_features.reset_index(drop = True)

    # 2. Data standardization
    scaler = StandardScaler()

    x_scaler_features = scaler.fit_transform(scaler_features)
    x_scaler_features = pd.DataFrame(x_scaler_features)
    # Merge the one - hot encoded columns with the original data
    x_scaler = pd.concat([x_scaler_features, BMPs_features], axis = 1)
    features_names = ['PCP', 'RH', 'MAXTM', 'MINTM', 'WIND', 'FSW', 'FSC', 'FRT_off', 'GW_0', 'GW_1']
    x_scaler.columns = features_names
    # Save the x scaler
    xscaler_name = 'xscaler' + sub_num + '.pkl'
    # Save the standardization model x
    pickle.dump(scaler, open(xscaler_name, 'wb'))

    # Perform 5 - fold cross - validation
    kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
    model = RandomForestRegressor(n_estimators = 10, criterion = 'friedman_mse', random_state = 1, n_jobs = -1)

    # Use cross_val_score for cross - validation. The core purpose is to evaluate the generalization ability of the model on different data subsets.
    cv_scores = cross_val_score(model, x_scaler, labels, cv = kf, scoring = 'r2')
    print("Cross - validation R^2 scores:", cv_scores)
    print("Mean cross - validation R^2 score:", np.mean(cv_scores))

    # 4. Split the dataset into training and test sets for final model training and evaluation, which is used for practical applications
    x_train, x_test, y_train, y_test = train_test_split(x_scaler, labels, test_size = 0.2, random_state = 42)

    # Model instantiation
    model = RandomForestRegressor(n_estimators = 10, criterion = 'friedman_mse', random_state = 1, n_jobs = -1)
    # Model training
    model.fit(x_train, y_train)
    # Performance evaluation on the test set
    r2_1 = model.score(x_test, y_test)
    YP1 = model.predict(x_train)
    YP1 = pd.DataFrame(YP1)
    # Performance evaluation on the training set
    r2 = r2_score(YP1, y_train)
    print("R^2 of the training set:", r2)
    print("R^2 of the validation set:", r2_1)
    # Save the model to a file
    model_name = 'SMFsub' + sub_num + '.pkl'
    joblib.dump(model, model_name)


# In[ ]:


# Set the simulation years
bg_time = 2015    # Start time of the simulation
end_time = 2022   # End time of the simulation

# Read the basic information of subwatershed
subdata = pd.read_csv("Basic_Information_of_subwatershed.csv")

# Define the subbasin_features(SUB, GW, FSW, FSC, FRT_off) function to generate data for a single sub - basin
def subbasin_features(SUB, GW, FSW, FSC, FRT_off):
    # Read the weather station corresponding to the sub - basin
    weather_station = subdata[subdata['SUB'] == SUB]['station']
    # Extract the integer from the read data
    station_num = weather_station.iloc[0]
    # Convert the number to a string
    station_num = str(station_num)
    # Read the weather data
    weather_data = pd.read_csv(station_num + ".csv")
    if FSW < 1:
        FSW = 0
        FSC = 0
    else:
        FSW = FSW
        FSC = FSC

    if GW > 0.5:
        GW_0 = 0
        GW_1 = 1
    else:
        GW_0 = 1
        GW_1 = 0

    # # Insert the parameters column
    weather_data.insert(0, 'SUB', SUB)
    weather_data.insert(9, 'FSW', FSW)
    weather_data.insert(10, 'FSC', FSC)
    weather_data.insert(11, 'FRT_off', FRT_off)
    # Insert the GW parameter
    weather_data.insert(12, 'GW_0', GW_0)
    weather_data.insert(13, 'GW_1', GW_1)
    # Create a filtering condition based on the simulation time
    sc = str(bg_time) + "<= YEAR <=" + str(end_time)
    # Filter the weather data
    filtered_data = weather_data.query(sc)
    filtered_data = filtered_data.drop(columns=['DATE', 'SUB', 'YEAR', 'MON'])
    features = filtered_data
    features = pd.DataFrame(features)
    return features

# Define the simulation function for the sub - basin surrogate model
def predict_model_TOT_N(features, SUB):
    SUB_num = "{:02}".format(SUB)

    loaded_xcaler_name = 'xscaler' + SUB_num + '.pkl'
    # Load the sub - basin surrogate model
    loaded_model = joblib.load(model_name)
    # Load the standardization model
    loaded_xscaler = pickle.load(open(loaded_xcaler_name, 'rb'))

    scaler_features = features[['PCP', 'RH', 'MAXTM', 'MINTM', 'WIND']]
    BMPs_features = features[['FSW', 'FSC', 'FRT_off', 'GW_0', 'GW_1']]
    # Reset the index
    BMPs_features = BMPs_features.reset_index(drop=True)
    xv_scaler_features = loaded_xscaler.transform(scaler_features)
    xv_scaler_features = pd.DataFrame(xv_scaler_features)
    xv_scaler = pd.concat([xv_scaler_features, BMPs_features], axis=1)
    features_names = ['PCP', 'RH', 'MAXTM', 'MINTM', 'WIND', 'FSW', 'FSC', 'FRT_off', 'GW_0', 'GW_1']
    xv_scaler.columns = features_names
    # Use the loaded model for prediction
    yv_pred = loaded_model.predict(xv_scaler)

    y_pre_out = pd.DataFrame(yv_pred)  # Convert to DataFrame
    # Remove negative values
    result = y_pre_out.clip(lower=0)

    predict_output = pd.DataFrame(result)
    column_WYLD = 'SUB_' + str(SUB) + '_WYLD'
    column_SYLD = 'SUB_' + str(SUB) + '_SYLD'
    column_N = 'SUB_' + str(SUB) + '_N'
    column_names = [column_WYLD, column_SYLD, column_N]
    # Modify the column names
    predict_output.columns = column_names
    sub_predict_output = predict_output
    return sub_predict_output


# In[11]:


# After defining the functions, you can call the sub - basin surrogate model as follows:
SUB = 1  # Input the number of the sub - basin to be called

# Input the BMPs parameters for simulation (see the main text of the paper for the parameters value ranges)
GW = 1  # Input the GW parameter
FSW = 15  # Input the FSW parameter
FSC = 60  # Input the FSC parameter
FRT_off = 10  # Input the FRT_off parameter

# Perform the simulation of the sub - basin surrogate model
sub_model_simulation = predict_model_TOT_N(subbasin_features(SUB, GW, FSW, FSC, FRT_off), SUB)

# sub_model_simulation contains the predicted water yield "WYLD", sediment load "SYLD", and total nitrogen load "N" at the subwatershed scale by the surrogate model,
# and its presentation format is consistent with that of the SWAT model.


# In[ ]:




