#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 07:50:19 2024

@author: sami_b
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgba


#tgt_building = 'Robin_education_Billi'
#
#src_building1 = 'Robin_education_Julius'
#src_building2 = 'Robin_education_Terrance'#
#src_building3 = 'Robin_education_Takako'
#src_building4 = 'Robin_education_Kristopher'


#Robin_education_Julius #src1
#Robin_education_Terrance #src2
#Robin_education_Takako #src3
#Robin_education_Kristopher #src4
#Robin_education_Billi #tgt
data=pd.read_csv('clean_genome_meters.csv', sep=',')

data_columns = {
    'Robin_education_Julius': 'S1',  # Replace with actual column name
    'Robin_education_Terrance': 'S2',  # Replace with actual column name
    'Robin_education_Takako': 'S3',  # Replace with actual column name
    'Robin_education_Kristopher': 'S4',  # Replace with actual column name
    'Robin_education_Billi': 'T',  # Replace with actual column name
}


# Define a base color
base_colors = ['blue','green','orange','pink','red']

# Create a figure
plt.figure(figsize=(8, 6))
i=0
# Loop through each dataset
for column in data_columns:
    if i<4:
        energy_consumption = data[column][:8784]
    
        # Adjust color brightness
        alpha = 0.4 + (i * 0.1)  # Increase alpha for brightness difference
        color = to_rgba(base_colors[i], alpha=alpha)
    
        # Plot histogram
        plt.hist(energy_consumption, bins=30, density=True, alpha=0.5, color=color)
    
        # Plot PDF using seaborn
        sns.kdeplot(energy_consumption, color=color, lw=2, label=f'{data_columns[column]}')
        i+=1
    else:
        energy_consumption = data[column][:1714]
        
        # Adjust color brightness
        alpha = 0.4 + (i * 0.1)  # Increase alpha for brightness difference
        color = to_rgba(base_colors[i], alpha=alpha)
        
        # Plot histogram
        plt.hist(energy_consumption, bins=30, density=True, alpha=0.5, color=color)
        
        # Plot PDF using seaborn
        sns.kdeplot(energy_consumption, color=color, lw=2, label=f'{data_columns[column]}')

# Add titles and labels
plt.title('Electricity load Histograms and KDEs')
plt.xlabel('Electricity load(KWh)')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig('histograms.png')
plt.show()


