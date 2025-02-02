#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 00:27:21 2024

@author: sami_b
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data_T=pd.read_csv('test_results_T_24h.csv')
data_1S=pd.read_csv('test_results_1S_24h.csv')
data_2S=pd.read_csv('test_results_2S_24h.csv')
data_3S=pd.read_csv('test_results_3S_24h.csv')
data_4S=pd.read_csv('test_results_4S_24h.csv')

# Sample data: Replace with your actual predictions and true values
y_true_T = data_T['y_true']  # Actual values
y_pred_T = data_T['y_hat']  # Predicted values
y_true_1S = data_1S['y_true']  # Actual values
y_pred_1S = data_1S['y_hat']  # Predicted values
y_true_2S = data_2S['y_true']  # Actual values
y_pred_2S = data_2S['y_hat']  # Predicted values
y_true_3S = data_3S['y_true']  # Actual values
y_pred_3S = data_3S['y_hat']  # Predicted values
y_true_4S = data_4S['y_true']  # Actual values
y_pred_4S = data_4S['y_hat']  # Predicted values

# Calculate the prediction errors
errors_T = y_pred_T-y_true_T
errors_1S= y_pred_1S-y_true_1S
errors_2S= y_pred_2S-y_true_2S
errors_3S= y_pred_3S-y_true_3S
errors_4S= y_pred_4S-y_true_4S



# Sample data: Replace these with your actual prediction errors
errors = {
    'T': errors_T,  # Example errors for Model A
    '1S': errors_1S,  # Example errors for Model B
    '2S': errors_2S,  # Example errors for Model C
    '3S': errors_3S,  # Example errors for Model D
    '4S': errors_4S,  # Example errors for Model E
}

# Create a DataFrame for plotting
error_data = []
for model, error_values in errors.items():
    for error in error_values:
        error_data.append((model, error))

error_df = pd.DataFrame(error_data, columns=['Model', 'Prediction Error'])

# Set up the plot
plt.figure(figsize=(8, 6))

# Use a vibrant color palette
palette = sns.color_palette("tab10")

for i, model in enumerate(errors.keys()):
    sns.kdeplot(
        data=error_df[error_df['Model'] == model],
        x='Prediction Error',
        color=palette[i],
        label=model,
        fill=False,
        lw=2.5  # Increased line width
    )

# Customize the plot
plt.title('Prediction Error Distribution for Different Models')
plt.xlabel('Prediction Error')
plt.ylabel('Density')
plt.axvline(0, color='red', linestyle='--')  # Optional: Line at y=0 for reference

# Create custom legend handles
legend_elements = [Patch(facecolor=sns.color_palette()[i], label=model) for i, model in enumerate(errors.keys())]
plt.legend(handles=legend_elements, title='Model')
# Save the figure
plt.savefig('prediction_error_distribution_24h.png', dpi=300, bbox_inches='tight')  # Save as PNG
plt.show()

########interpretration
# the best performance is achieved by 3S & 4S with thecloset error dstributions to the
#normal distribution
#the othre models 1S, 2S & T are more underpredicting the electricity load (more negative
#error than positive one)