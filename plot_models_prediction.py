#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 00:48:33 2024

@author: sami_b
"""

import pandas as pd
import matplotlib.pyplot as plt


tgt_horizon=1 #24 for 24h, 1 for 1h 

#297=len(y_true) for 24h
#test_time for 24h is 312 length
#y_true for 24h is 368 length
test_time=pd.to_datetime(pd.read_csv('test_time.csv')['timestamp'])#[-368:]#remove _24h for 1h


y_true=pd.read_csv('test_results_T_1h.csv')['y_true'][::tgt_horizon][-len(test_time):] #T_1h for 1h, 
y_hat_T=pd.read_csv('test_results_T_1h.csv')['y_hat'][::tgt_horizon][-len(test_time):]
y_hat_1S=pd.read_csv('test_results_1S_1h.csv')['y_hat'][::tgt_horizon][-len(test_time):]
y_hat_2S=pd.read_csv('test_results_2S_1h.csv')['y_hat'][::tgt_horizon][-len(test_time):]
y_hat_3S=pd.read_csv('test_results_3S_1h.csv')['y_hat'][::tgt_horizon][-len(test_time):]
y_hat_4S=pd.read_csv('test_results_4S_1h.csv')['y_hat'][::tgt_horizon][-len(test_time):]



# Plotting
plt.figure(figsize=(8, 6))

# Plot true load values
plt.plot(test_time, y_true, linestyle='--', color='black', label='True', linewidth=2)

# Plot predictions from each model with different colors
plt.plot(test_time, y_hat_T, color='blue', label='T')
plt.plot(test_time, y_hat_1S, color='orange', label='1S')
plt.plot(test_time, y_hat_2S, color='green', label='2S')
plt.plot(test_time, y_hat_3S, color='red', label='3S')
plt.plot(test_time, y_hat_4S, color='purple', label='4S')



# Adding labels and title
plt.xlabel('Time')
plt.ylabel('Load Values(KWh)')
plt.title('True Load vs. Model Predictions')
# Place the legend outside the plot
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)
plt.grid()

# Show the plot
plt.tight_layout()
# Save the plot
plt.savefig('load_predictions_'+str(tgt_horizon)+'h.png', bbox_inches='tight') 
plt.show()


