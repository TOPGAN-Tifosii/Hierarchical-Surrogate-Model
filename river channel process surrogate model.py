#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import tensorflow.keras.backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow.keras import Sequential, layers, losses, utils
import joblib


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Generate training data for the river channel model
SUB_OUT = []
RCH_OUT = []

# Please enter the number of subbasins at “XX” below,make sure that the training data for each sub-watershed(subDD_xx.csv) has been prepared.
NOS = XX

for i in range(1, NOS + 1):
    file_name = f'subDD_{i:02d}.csv'
    sub_num = i - 1
    dataset = pd.read_csv(file_name)
    # Extract sub - basin relevant data
    SUB_DATA = dataset[['WYLDmm', 'SYLDt_ha', 'subTN']]
    sub_columns_name = ['SUB_' + str(i) + '_WYLD', 'SUB_' + str(i) + '_SYLD', 'SUB_' + str(i) + '_N']
    rch_columns_name = ['FLOW_out_' + str(i), 'SED_out_' + str(i), 'TOT_N_' + str(i)]
    SUB_DATA.columns = sub_columns_name

    # Extract river channel relevant data
    RCH_DATA = dataset[['FLOW_OUTcms', 'SED_OUTtons', 'TOT_Nkg']]
    RCH_DATA.columns = rch_columns_name

    SUB_OUT.append(SUB_DATA)
    RCH_OUT.append(RCH_DATA)


SUB_OUT_F = pd.concat(SUB_OUT, axis=1)
RCH_OUT_F = pd.concat(RCH_OUT, axis=1)
outputdata = pd.concat([SUB_OUT_F, RCH_OUT_F], axis=1)
# Save the generated training data for the river channel process model
outputdata.to_csv("RCH_DATA.txt", index=False)


# In[ ]:


# Before starting the model training,
#please create the "RCH_DATA.txt" file as the training data for the river channel process surrogate model according to the instructions.

# Load the dataset
dataset = pd.read_csv("RCH_DATA.txt", sep=',')

# Separate the feature set and the label set
x = dataset.iloc[:, :87]
y = dataset.iloc[:, -87:]

xscaler = StandardScaler()
yscaler = StandardScaler()

# Data standardization
x_scaler = xscaler.fit_transform(x)
y_scaler = yscaler.fit_transform(y)

# Save the scalers
pickle.dump(xscaler, open('xscaler_rch.pkl', 'wb'))  # Save the standardization model for x
pickle.dump(yscaler, open('yscaler_rch.pkl', 'wb'))  # Save the standardization model for y

# Split the dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x_scaler, y_scaler, test_size=0.2, random_state=666)


# In[ ]:


# Define the coefficient of determination(R^2) as the evaluation criterion
def r_squared(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))
    
# Hyperparameter settings for the ANN algorithm

# Number of training epochs
EPOCHS = 100

BATCH_SIZE = 32

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Visualization of the results (Coefficient of determination R^2)
def plot_results(history):
    '''
    history: The training results returned after the model training is completed.
    '''
    # Display the coefficient of determination (R^2)
    plt.figure(figsize=(10, 6))
    epoch_range = range(1, len(history.history['r_squared']) + 1)
    plt.plot(epoch_range, history.history['r_squared'], label='train R^2')
    plt.plot(epoch_range, history.history['val_r_squared'], label='val R^2')
    plt.title("Coefficient of determination (R^2)")
    plt.xlabel("Epoch")
    plt.ylabel("R^2")
    plt.legend(loc='best')
    plt.show()
    
    # Display the LOSS
    plt.figure(figsize=(10,6)) # Canvas size
    plt.plot(epoch_range, history.history['loss'], label='train loss')
    plt.plot(epoch_range, history.history['val_loss'], label='val loss')
    plt.title('LOSS')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.show()

# Basic model settings
model_2 = Sequential([
    # First layer
    layers.Dense(units=126,
                 activation='relu',
                 input_shape=[63]), # Each sub - basin has 3 input features. In this study, there are 21 sub - basins, so the input dimension is 63.
    
    # Second layer
    layers.Dense(units=252,
                 activation='relu'),
    
    # Third layer
    layers.Dense(units=126,
                 activation='relu'),

    # Fourth layer
    layers.Dense(units=126,
                 activation='relu'),
    
    layers.Dense(63)
])

# Model compilation
model_2.compile(optimizer = 'adam',
              loss= 'mse',
              metrics=[r_squared])

# Model training
history_2 = model_2.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, y_test), verbose=1)

# Display the results
plot_results(history_2)


# In[ ]:


# Model prediction and model performance presentation
y_pred = model_2.predict(x_scaler)

# Inverse standardization of the prediction results
y_pred_inv = yscaler.inverse_transform(y_pred)
y_pred_inv = y_pred_inv.clip(min=0)  # Remove negative values

r2 = r2_score(y, y_pred_inv, multioutput='raw_values')

# Group the r2 values in the specified order
FLOW = r2[::3]
SED = r2[1::3]
TN = r2[2::3]

# Calculate the maximum value
max1 = np.max(FLOW)
print("Maximum value of FLOW:", max1)
# Calculate the minimum value
min1 = np.min(FLOW)
print("Minimum value of FLOW:", min1)
# Calculate the mean value
mean = np.mean(FLOW)
print("Mean value of FLOW:", mean)
# Calculate the median value
median = np.median(FLOW)
print("Median value of FLOW:", median)

# Calculate the maximum value
max1 = np.max(SED)
print("Maximum value of SED:", max1)
# Calculate the minimum value
min1 = np.min(SED)
print("Minimum value of SED:", min1)
# Calculate the mean value
mean = np.mean(SED)
print("Mean value of SED:", mean)
# Calculate the median value
median = np.median(SED)
print("Median value of SED:", median)

# Calculate the maximum value
max1 = np.max(TN)
print("Maximum value of TN:", max1)
# Calculate the minimum value
min1 = np.min(TN)
print("Minimum value of TN:", min1)
# Calculate the mean value
mean = np.mean(TN)
print("Mean value of TN:", mean)
# Calculate the median value
median = np.median(TN)
print("Median value of TN:", median)

# Plot the probability distribution of FLOW
sns.distplot(FLOW)
plt.title("FLOW_R^2")
plt.show()

# Plot the probability distribution of SED
sns.distplot(SED)
plt.title("SED_R^2")
plt.show()

# Plot the probability distribution of TN
sns.distplot(TN)
plt.title("TN_R^2")
plt.show()


# In[ ]:


#save river channel process surrogate model
model_2.save('RCH_SM.h5')

