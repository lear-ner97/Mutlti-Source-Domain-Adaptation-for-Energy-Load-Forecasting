#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 09:20:08 2025

@author: sami_b
"""

#Note: 
#you have to complete the running of the main file, then you run this file

import matplotlib.pyplot as plt
import numpy as np
import torch
import functions

#full_model=functions.full_model(kernel_size,stride,in_channels0,out_channels0,in_channels1,out_channels1,input_size, num_gru_units, hidden_size, num_gru_layers,hidden_size,hidden_size,forecast_length).to(device)
#full_model_state_dict = torch.load(best_full_model_path)
#full_model.load_state_dict(full_model_state_dict)
##########################################    
tcn_predicted = full_model.feature_extractor.tcn(X_train_tgt.to(device))  # [0]
bigru_predicted = full_model.feature_extractor.bigru(X_train_tgt.to(device))
predicted_features=tcn_predicted+bigru_predicted
attention_weights=full_model.feature_extractor.attn(predicted_features)[1].squeeze()

feature_names = tgt_columns[:-tgt_horizon]
# Visualize the attention weights as a heatmap
# Select every nth feature (e.g., every 5th feature)
step = 10 # 10 if H=24, 1 if H=1
subset_feature_names = feature_names[::step]  # Take every 5th feature

# Adjust the figure size
plt.figure()
plt.imshow(attention_weights.cpu().detach().numpy(), cmap='viridis', aspect='auto')

# Add color bar
plt.colorbar(label='Attention Weight')

# Add title and labels
plt.title('Attention Weights Heatmap')
plt.xlabel('Feature Name')
plt.ylabel('Sample Index')

# Set x-ticks to be the subset of feature names
plt.xticks(ticks=range(0, len(feature_names), step), labels=subset_feature_names, rotation=90, fontsize=8)

# Adjust the layout to ensure labels don't overlap
plt.tight_layout()

# Save and display the heatmap
plt.savefig('attention_weights_heatmap.png', bbox_inches='tight')
plt.show()




average_attention = attention_weights.mean(dim=0)

# Convert average attention weights to numpy array for plotting
average_weight_per_feature = average_attention.cpu().detach().numpy()

# Create a subset of feature names (every 5th feature name)
subset_feature_names = tgt_columns[:-tgt_horizon][::step]  # Excluding last 24 features if needed and taking every 5th feature

# Plot the average attention weights across features
plt.figure()
plt.bar(range(len(subset_feature_names)), average_weight_per_feature[::step])

# Add title and labels
plt.title('Average Attention Weights Across All Samples')
plt.xlabel('Feature Name')
plt.ylabel('Average Attention Weight')

# Set x-ticks to be the subset of feature names
plt.xticks(ticks=range(len(subset_feature_names)), labels=subset_feature_names, rotation=90, fontsize=8)

# Adjust the layout to prevent label overlap
plt.tight_layout()

# Save and display the plot
plt.savefig('average_attention_weights.png', bbox_inches='tight')
plt.show()

# Get the indices of the top 3 highest values
top_n = 10
# Get indices of top_n features with the highest average attention weights
top_n_indices = np.argsort(average_weight_per_feature)[-top_n:][::-1]

# Get the values corresponding to the top_n indices
top_n_values = average_weight_per_feature[top_n_indices]

# Get the names of the top_n features
top_n_feature_names = np.array(tgt_columns[:-tgt_horizon])[top_n_indices]  # Exclude last 24 features if necessary

# Print top n highest values and their corresponding feature names
print(f"Top {top_n} highest values and their feature names:")
for feature, value in zip(top_n_feature_names, top_n_values):
    print(f"Feature: {feature}, Value: {value}")

# Plot the top_n average attention weights across features
plt.figure(figsize=(10, 6))
plt.barh(top_n_feature_names, top_n_values, color='skyblue')

# Add title and labels
plt.title(f'Top {top_n} Most Important Features Based on Average Attention Weights')
plt.xlabel('Average Attention Weight')
plt.ylabel('Feature Name')

# Adjust the layout to prevent label overlap
plt.tight_layout()

# Save and display the plot
plt.savefig('top_n_average_attention_weights.png', bbox_inches='tight')
plt.show()