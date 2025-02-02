#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 09:49:49 2025

@author: sami_b
"""
#Note: 
#you have to complete the running of the main file, then you run this file

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from functions import *
from sklearn.preprocessing import StandardScaler
import shap
from shap import DeepExplainer,KernelExplainer,GradientExplainer



# You can also use a random subset from the training set to get a balanced representation
background_data = X_train_tgt.to(device)#[np.random.choice(X_train_tgt.shape[0], size=500, replace=False)].to(device)
#using all the training data as background data for a better fit of the explainer
#to the data distribution
# Initialize the SHAP explainer, here we use gradient explainer
#DeepExplainer gave an error of threshold error
explainer = GradientExplainer(full_model, background_data)
#the pred function of the model has to return a single output not a tuple
full_model.train(True) 
#compute the shap values

#number of samples to explain
nbr_samples_to_explain=100
# Get SHAP values for a sample from your test data
# the test data time period is in January and february, so choosing any 100
# random points will be convenient
#explainer.shap_values
shap_values = explainer.shap_values(X_test_tgt[:nbr_samples_to_explain].to(device))  # X_test is the data you want to explain
#the more data you explain the broader and more generalizing is your explanation
#the smaller the data size the more local is the explanation
#shap.plots.bar(shap_values)

feature_names=['cosine_transform_hour_of_day','is_holiday','is_weekend','airTemperature','windSpeed']+[f"Lag {i}" for i in range(src_lookback,0,-1)]

avg_shap_values = np.mean(np.array(shap_values), axis=0).squeeze(axis=-1) 

shap.summary_plot(avg_shap_values, X_test_tgt[:nbr_samples_to_explain].cpu().numpy().squeeze(), feature_names=feature_names,max_display=50)
#plot the bar plot
shap.summary_plot(avg_shap_values, X_test_tgt[:nbr_samples_to_explain], feature_names=feature_names, plot_type="bar")#,max_display=12)
#dependance plot
shap.dependence_plot(0, avg_shap_values, X_test_tgt[:nbr_samples_to_explain].cpu().numpy().squeeze(), feature_names=feature_names,  show=True,interaction_index=3)#,with temperature
#plt.show()
for horizon_idx in range(7,8):#src_horizon  # Loop through the 24 forecast horizons
    #comment the next line when H=1
    shap_vals_at_horizon = shap_values[horizon_idx]  # Shape: (num_samples, num_time_steps, 1)
    
    # Squeeze the last dimension to get rid of the single value (it’s not needed for plotting)
    shap_vals_at_horizon = shap_vals_at_horizon.squeeze(axis=-1)  # Now shape: (num_samples, num_time_steps)
    #shap_vals_at_horizon in the last line
    #shap.plots.bar(shap_values)
    # Create a summary plot for this specific forecast horizon
    print(f"Summary plot for forecast horizon {horizon_idx + 1}")
    #plot the summary plot
    shap.summary_plot(shap_vals_at_horizon, X_test_tgt[:nbr_samples_to_explain].cpu().numpy().squeeze(), feature_names=feature_names)
    #plot the bar plot
    shap.summary_plot(shap_vals_at_horizon, X_test_tgt[:nbr_samples_to_explain], feature_names=feature_names, plot_type="bar",max_display=12)
    #dependance plot
    shap.dependence_plot(0, shap_vals_at_horizon, X_test_tgt[:nbr_samples_to_explain].cpu().numpy().squeeze(), feature_names=feature_names,  show=True,interaction_index=3)#,with temperature
    plt.show()
#    ['cosine_transform_hour_of_day', 'is_holiday', 'is_weekend', 'airTemperature', 'windSpeed']
#     Plot SHAP interaction plot for two features
#    index 0 is for cosine transform
    #################################



#Features (time steps) are sorted by importance: By default, the summary plot orders 
#the time steps by the average magnitude of their Shapley values across the samples. 
#This means that the most important time steps (i.e., those contributing the most to 
#the model's prediction) will appear first in the plot.
#
#The color gradient in the plot will represent the magnitude of the Shapley values, 
#where positive contributions will push the prediction up, and negative ones will 
#push it down.
