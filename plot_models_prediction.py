#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 00:48:33 2024

@author: sami_b
"""

import pandas as pd
import matplotlib.pyplot as plt


tgt_horizon=24 #24 for 24h, 1 for 1h 

#297=len(y_true) for 24h
#test_time for 24h is 312 length
#y_true for 24h is 368 length
folder='prediction_results/'
test_time=pd.to_datetime(pd.read_csv(folder+'test_time.csv')['timestamp'])#[-24:]#remove _24h for 1h


y_true=pd.read_csv(folder+f'test_results_T_{tgt_horizon}h.csv')['y_true'][::tgt_horizon][-len(test_time):] #T_1h for 1h, 
y_hat_T=pd.read_csv(folder+f'test_results_T_{tgt_horizon}h.csv')['y_hat'][::tgt_horizon][-len(test_time):]
y_hat_1S=pd.read_csv(folder+f'test_results_1S_{tgt_horizon}h.csv')['y_hat'][::tgt_horizon][-len(test_time):]
y_hat_2S=pd.read_csv(folder+f'test_results_2S_{tgt_horizon}h.csv')['y_hat'][::tgt_horizon][-len(test_time):]
y_hat_3S=pd.read_csv(folder+f'test_results_3S_{tgt_horizon}h.csv')['y_hat'][::tgt_horizon][-len(test_time):]
y_hat_4S=pd.read_csv(folder+f'test_results_4S_{tgt_horizon}h.csv')['y_hat'][::tgt_horizon][-len(test_time):]




# Plotting
plt.figure(figsize=(10, 6))  # Increase figure width

# Plot true load values with increased line width and solid style
plt.plot(test_time, y_true, linestyle='-', color='black', label='True', linewidth=3)  # Thicker line for 'True'

# Plot predictions from each model with different colors and line styles
plt.plot(test_time, y_hat_T, color='blue', label='T', linestyle='--', linewidth=2)
plt.plot(test_time, y_hat_1S, color='orange', label='1S', linestyle='-.', linewidth=2)
plt.plot(test_time, y_hat_2S, color='green', label='2S', linestyle=':', linewidth=2)
plt.plot(test_time, y_hat_3S, color='red', label='3S', linestyle='-', linewidth=2)
plt.plot(test_time, y_hat_4S, color='purple', label='4S', linestyle='-', linewidth=2)

# Adding labels and title
plt.xlabel('Time', fontsize=12)  # Increase font size
plt.ylabel('Load (kWh)', fontsize=12)
#plt.title('True Load vs Model Predictions', fontsize=14)

# Place the legend outside the plot for better visibility
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0., fontsize=12)

# Make gridlines less dominant but still present
plt.grid(True, linestyle='--', alpha=0.6)

# Adjust plot layout to ensure everything fits
plt.tight_layout()

# Save the plot with a more descriptive filename
plt.savefig('load_predictions_'+str(tgt_horizon)+'h_comparison.png', bbox_inches='tight')

# Show the plot
plt.show()



