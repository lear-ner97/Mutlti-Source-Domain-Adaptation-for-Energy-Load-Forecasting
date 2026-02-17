#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 09:49:49 2025

@author: sami_b
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from functions import *
from sklearn.preprocessing import StandardScaler
import shap
from shap import DeepExplainer,KernelExplainer,GradientExplainer





################## IMPORTANT
#Note: 
#you have to complete the running of the main file  main_multi_source_implementation.py, then you run this file 
#full_model is an instance of the validated model in main_multi_source_implementation.py




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


############################# TASK1: compute global relative feature importance 
#1-Global Feature Importance (Mean Absolute SHAP)
mean_abs_shap = np.mean(np.abs(shap_values), axis=(0,3))#axis=0 for hourly, (0,3) for daily
#the order of features in target is as follows:
features = ['cosine_transform_hour_of_day', 'is_holiday', 'is_weekend', 'airTemperature','windSpeed'] + [f"load_lag{i}" for i in range(tgt_lookback+tgt_horizon-1, 0, -1)]#+[tgt_building]
mean_abs_shap_flat = mean_abs_shap.squeeze()  # shape (len(features),)

# Optional: sort by importance
limit_features=29 #for daily forecasting only because there are 173 features
sorted_idx = np.argsort(mean_abs_shap_flat)[::-1]  # descending order
sorted_features = [features[i] for i in sorted_idx]
sorted_values = mean_abs_shap_flat[sorted_idx]
# Plot horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(sorted_features[:limit_features][::-1], sorted_values[:limit_features][::-1], color="skyblue")
plt.xlabel("Mean |SHAP Value|")
#plt.title("Feature Importance (Top Features)")
max_val = sorted_values.max()
plt.xlim(0, max_val * 1.1)  # 5% extra space
plt.tight_layout()
plt.savefig('mean_abs_shap.pdf', format='pdf')
plt.show()


#sorted (column/feature wise) absolute shap values 
#abs_shap_values = np.abs(shap_values)[:,sorted_idx]  # element-wise absolute (hourly)
abs_shap_values = np.mean(np.abs(shap_values)[:,sorted_idx],axis=-1)  #daily
#apply wilcoxon
from scipy.stats import wilcoxon
shap_feature1 = abs_shap_values[:, 2]
shap_feature2 = abs_shap_values[:, 4]
stat, p_value = wilcoxon(shap_feature1, shap_feature2)

print(f"Wilcoxon statistic: {stat}, p-value: {p_value}")
##################################




##################### Task 2: show summary plots
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
