#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 17:26:55 2024

@author: sami_b
"""


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# Sample data: Replace this with your actual RMSE values
model_names = ['T', '1S', '2S', '3S', '4S']
mape_values = {
    'T': [10.419637497828814, 20.985431844525973, 13.517646204706251, 13.804841171522488, 9.983434039337462, 14.182531388768666, 19.047227610722434, 11.938270533850533, 13.04951015546297, 25.72191009735279],
    '1S': [8.70560373765283, 7.799099925355701, 8.356520887775492, 7.92382219018798, 8.80865643238766, 9.124816530919613, 9.035300738692452, 7.790304544612496, 7.846172223345686, 9.541049410274614],
    '2S': [7.464587688479146, 7.521745467716838, 7.433656191189028, 7.3201267496312825, 7.428619334466735, 7.0629187487604606, 7.217468617271646, 7.876015406761651, 6.666091930733281, 7.3934547059334665],
    '3S': [7.253848282539755, 6.636114824350203, 7.172477609066647, 7.0112877500067645, 6.9826032878027355, 7.6933254617578815, 7.31771206242737, 7.51761368947317, 6.8700393522919105, 7.584903571473113],
    '4S': [7.416564298882608, 7.048980930359153, 7.374904877144403, 7.014171233063536, 7.274205120911775, 6.9745349983966545, 7.8020200099279, 7.454730907520014, 6.797773767841169, 7.703204063673752] ,
}

# Convert the dictionary to a format suitable for Seaborn
data = []
for model, values in mape_values.items():
    for value in values:
        data.append((model, value))

# Create a DataFrame from the data
df = pd.DataFrame(data, columns=['Model', 'MAPE'])

# Set up the boxplot with a custom color palette
plt.figure(figsize=(8, 6))
palette = sns.color_palette("husl", len(model_names))  # Choose a color palette
sns.boxplot(x='Model', y='MAPE', data=df, palette=palette)
plt.title('Boxplots of MAPE Values for Different Models')
plt.xlabel('Model')
plt.ylabel('MAPE')
# Create custom legend
legend_elements = [Patch(facecolor=palette[i], label=model_names[i]) for i in range(len(model_names))]
plt.legend(handles=legend_elements, title='Models', loc='upper right')
# Save the figure as a PNG file
plt.savefig('boxplot_mape_1h.png', format='png', dpi=300, bbox_inches='tight')
# Show the plot
plt.show()
