#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pymoo.core.problem import ElementwiseProblem
import numpy as np
import torch.nn as nn
from torch.nn import Sequential
from torch.nn import Conv1d,Linear
import tensorflow.keras.backend as K
import torch
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from multiprocessing.pool import ThreadPool
from pymoo.core.problem import StarmapParallelization
from pymoo.algorithms.soo.nonconvex.ga import GA
from sklearn.preprocessing import StandardScaler
from pymoo.operators.sampling.lhs import LHS

from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Basic settings and surrogate models import

bg_time = 2015    # Start time of the simulation
end_time = 2022   # End time of the simulation

# Read the basic information of subwatersheds
subdata = pd.read_csv("Basic_Information_of_subwatershed.csv")

# Please enter the number of subwatersheds at “XX” below
NOS = XX

# Load the subwatershed surrogate models
sub_model = []
for SUB in range(1, NOS + 1):
    sub_num = "{:02}".format(SUB)
    model_name = 'SMFsub' + sub_num + '.pkl'
    sub_model.append(joblib.load(model_name))

# Load the river channel process model
loaded_rch_model = joblib.load("RCH_SM.h5")  # Load the river channel model

# Load the data standardization models
xscaler = pickle.load(open('xscaler_rch.pkl', 'rb'))
yscaler = pickle.load(open('yscaler_rch.pkl', 'rb'))


# In[ ]:


# Define the subbasin_features(SUB, GW, FSW, FSC, FRT_off) function to generate data for a single subwatershed
def subbasin_features(SUB, GW, FSW, FSC, FRT_off):
    # Read the weather station corresponding to the subwatershed
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

    # Insert the parameters column
    weather_data.insert(0, 'SUB', SUB)
    weather_data.insert(9, 'FSW', FSW)
    weather_data.insert(10, 'FSC', FSC)
    weather_data.insert(11, 'FRT_off', FRT_off)
    # Insert the GW parameter
    weather_data.insert(12, 'GW_0', GW_0)
    # Insert the GW parameter
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

    loaded_xcaler_name = 'xscaler' + SUB_num+ '.pkl'
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
    SUB_c = SUB -1
    yv_pred = sub_model[SUB_c].predict(xv_scaler)

    y_pre_out = pd.DataFrame(yv_pred)  # Convert to DataFrame
   
    result = y_pre_out.clip(lower=0) # Remove negative values

    predict_output = pd.DataFrame(result)
    column_WYLD = 'SUB_' + str(SUB) + '_WYLD'
    column_SYLD = 'SUB_' + str(SUB) + '_SYLD'
    column_N = 'SUB_' + str(SUB) + '_N'
    column_names = [column_WYLD, column_SYLD, column_N]
    # Modify the column names
    predict_output.columns = column_names
    sub_predict_output = predict_output
    return sub_predict_output


# In[ ]:


# Define the invocation function for the river channel process surrogate model

# Calculate the simulation results at the scale of each sub - basin by invoking the sub - basin surrogate model
# All surrogate models need to be prepared before starting
def test_rch_features(GW_1, FSW_1, FSC_1, FRT_OFF_1,
                      GW_2, FSW_2, FSC_2, FRT_OFF_2,
                      GW_3, FSW_3, FSC_3, FRT_OFF_3,
                      GW_4, FSW_4, FSC_4, FRT_OFF_4,
                      GW_5, FSW_5, FSC_5, FRT_OFF_5,
                      GW_6, FSW_6, FSC_6, FRT_OFF_6,
                      GW_7, FSW_7, FSC_7, FRT_OFF_7,
                      GW_8, FSW_8, FSC_8, FRT_OFF_8,
                      GW_9, FSW_9, FSC_9, FRT_OFF_9,
                      GW_10, FSW_10, FSC_10, FRT_OFF_10,
                      GW_11, FSW_11, FSC_11, FRT_OFF_11,
                      GW_12, FSW_12, FSC_12, FRT_OFF_12,
                      GW_13, FSW_13, FSC_13, FRT_OFF_13,
                      GW_14, FSW_14, FSC_14, FRT_OFF_14,
                      GW_15, FSW_15, FSC_15, FRT_OFF_15,
                      GW_16, FSW_16, FSC_16, FRT_OFF_16,
                      GW_17, FSW_17, FSC_17, FRT_OFF_17,
                      GW_18, FSW_18, FSC_18, FRT_OFF_18,
                      GW_19, FSW_19, FSC_19, FRT_OFF_19,
                      GW_20, FSW_20, FSC_20, FRT_OFF_20,
                      GW_21, FSW_21, FSC_21, FRT_OFF_21):

    test_rch_fr = pd.DataFrame()

    GW = [GW_1, GW_2, GW_3, GW_4, GW_5, GW_6, GW_7, GW_8, GW_9, GW_10, GW_11, GW_12, GW_13, GW_14, GW_15, GW_16,
          GW_17, GW_18, GW_19, GW_20, GW_21]

    FSW = [FSW_1, FSW_2, FSW_3, FSW_4, FSW_5, FSW_6, FSW_7, FSW_8, FSW_9, FSW_10, FSW_11, FSW_12, FSW_13, FSW_14, FSW_15, FSW_16,
           FSW_17, FSW_18, FSW_19, FSW_20, FSW_21]

    FSC = [FSC_1, FSC_2, FSC_3, FSC_4, FSC_5, FSC_6, FSC_7, FSC_8, FSC_9, FSC_10, FSC_11, FSC_12, FSC_13, FSC_14, FSC_15, FSC_16,
           FSC_17, FSC_18, FSC_19, FSC_20, FSC_21]

    FRT_off = [FRT_OFF_1, FRT_OFF_2, FRT_OFF_3, FRT_OFF_4, FRT_OFF_5, FRT_OFF_6, FRT_OFF_7, FRT_OFF_8, FRT_OFF_9, FRT_OFF_10,
               FRT_OFF_11, FRT_OFF_12, FRT_OFF_13, FRT_OFF_14, FRT_OFF_15, FRT_OFF_16, FRT_OFF_17, FRT_OFF_18, FRT_OFF_19,
               FRT_OFF_20, FRT_OFF_21]
    # Use a for loop to iterate over the values of SUB
    for SUB in range(1, NOS + 1):
        sub1 = SUB - 1
        # Invoke the predict_model_TOT_N function and append the results to a new DataFrame
        extracted_data = predict_model_TOT_N(subbasin_features(SUB, GW[sub1], FSW[sub1], FSC[sub1], FRT_off[sub1]), SUB)
        test_rch_fr = pd.concat([test_rch_fr, extracted_data], axis=1)

    return test_rch_fr

# Define the prediction process for the river channel process surrogate model
def rch_predict_TN(test_rch_fr):

    features_scaler = xscaler.transform(test_rch_fr)
    # Model prediction
    y_pred_out = loaded_rch_model.predict(features_scaler, verbose=0)

    # Inverse standardization of the prediction results
    y_pre_out = yscaler.inverse_transform(y_pred_out)
    # Add column headers
    rch_predict_TN = pd.DataFrame(y_pre_out, columns=["FLOW_out_1", "SED_out_1", "TOT_N_1",
                                                      "FLOW_out_2", "SED_out_2", "TOT_N_2",
                                                      "FLOW_out_3", "SED_out_3", "TOT_N_3",
                                                      "FLOW_out_4", "SED_out_4", "TOT_N_4",
                                                      "FLOW_out_5", "SED_out_5", "TOT_N_5",
                                                      "FLOW_out_6", "SED_out_6", "TOT_N_6",
                                                      "FLOW_out_7", "SED_out_7", "TOT_N_7",
                                                      "FLOW_out_8", "SED_out_8", "TOT_N_8",
                                                      "FLOW_out_9", "SED_out_9", "TOT_N_9",
                                                      "FLOW_out_10", "SED_out_10", "TOT_N_10",
                                                      "FLOW_out_11", "SED_out_11", "TOT_N_11",
                                                      "FLOW_out_12", "SED_out_12", "TOT_N_12",
                                                      "FLOW_out_13", "SED_out_13", "TOT_N_13",
                                                      "FLOW_out_14", "SED_out_14", "TOT_N_14",
                                                      "FLOW_out_15", "SED_out_15", "TOT_N_15",
                                                      "FLOW_out_16", "SED_out_16", "TOT_N_16",
                                                      "FLOW_out_17", "SED_out_17", "TOT_N_17",
                                                      "FLOW_out_18", "SED_out_18", "TOT_N_18",
                                                      "FLOW_out_19", "SED_out_19", "TOT_N_19",
                                                      "FLOW_out_20", "SED_out_20", "TOT_N_20",
                                                      "FLOW_out_21", "SED_out_21", "TOT_N_21"])
    # Set the section number for data extraction
    rch64TN = rch_predict_TN["TOT_N_18"].astype(np.float64)
    rch64TN = rch64TN.clip(lower=0)  # Remove negative values
    rch_TN_sum = rch64TN.sum()

    return rch_TN_sum


# In[ ]:


# Assign values to GW, FSW, FSC, and FRT_OFF for each corresponding sub - basin to invoke the hierarchical surrogate model for watershed simulation
# The simulation data of the desired cross - section can be obtained by modifying rch_predict_TN(test_rch_fr)

model_simulation = rch_predict_TN(test_rch_features(GW_1, FSW_1, FSC_1, FRT_OFF_1,
                      GW_2, FSW_2, FSC_2, FRT_OFF_2,
                      GW_3, FSW_3, FSC_3, FRT_OFF_3,
                      GW_4, FSW_4, FSC_4, FRT_OFF_4,
                      GW_5, FSW_5, FSC_5, FRT_OFF_5,
                      GW_6, FSW_6, FSC_6, FRT_OFF_6,
                      GW_7, FSW_7, FSC_7, FRT_OFF_7,
                      GW_8, FSW_8, FSC_8, FRT_OFF_8,
                      GW_9, FSW_9, FSC_9, FRT_OFF_9,
                      GW_10, FSW_10, FSC_10, FRT_OFF_10,
                      GW_11, FSW_11, FSC_11, FRT_OFF_11,
                      GW_12, FSW_12, FSC_12, FRT_OFF_12,
                      GW_13, FSW_13, FSC_13, FRT_OFF_13,
                      GW_14, FSW_14, FSC_14, FRT_OFF_14,
                      GW_15, FSW_15, FSC_15, FRT_OFF_15,
                      GW_16, FSW_16, FSC_16, FRT_OFF_16,
                      GW_17, FSW_17, FSC_17, FRT_OFF_17,
                      GW_18, FSW_18, FSC_18, FRT_OFF_18,
                      GW_19, FSW_19, FSC_19, FRT_OFF_19,
                      GW_20, FSW_20, FSC_20, FRT_OFF_20,
                      GW_21, FSW_21, FSC_21, FRT_OFF_21))

