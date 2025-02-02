#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 17:33:26 2024

@author: sami_b
"""


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
r2score_values = {
    'T': [0.9269743440587043, 0.9082113393902284, 0.9229474171225579, 0.9305894922437753, 0.8976307139370521, 0.9147580883533921, 0.9037918632526911, 0.9005005308021943, 0.9308700144091592, 0.9089291932328565],
    '1S':  [0.9340322131145339, 0.9268491666616097, 0.9176540996892012, 0.9297066025982902, 0.916239427962298, 0.9267646869882592, 0.9159353055474839, 0.9219942359092865, 0.9270435968693381, 0.9197135993806477],
    '2S': [0.9213567227277736, 0.9205671092633886, 0.9014536974847247, 0.900120223156435, 0.9188320835492313, 0.9161786520886398, 0.9218749442802876, 0.922275018245646, 0.9271026716492651, 0.9288996000686502],
    '3S': [0.9389025639480555, 0.9124302037988002, 0.9063143882030671, 0.9089324104220541, 0.9169094791009543, 0.9258835954461849, 0.9358288580629868, 0.9216051680935674, 0.931034058811397, 0.9241819916844355],
    '4S':  [0.9043391462889866, 0.9048928995072302, 0.899697194431519, 0.9073243799236234, 0.9146841449990933, 0.9240530691166478, 0.9212313990152877, 0.9185850773080708, 0.9266125304053642, 0.9012820178024371],
}

# Convert the dictionary to a format suitable for Seaborn
data = []
for model, values in r2score_values.items():
    for value in values:
        data.append((model, value))

# Create a DataFrame from the data
df = pd.DataFrame(data, columns=['Model', 'R2-score'])

# Set up the boxplot with a custom color palette
plt.figure(figsize=(8, 6))
palette = sns.color_palette("husl", len(model_names))  # Choose a color palette
sns.boxplot(x='Model', y='R2-score', data=df, palette=palette)
plt.title('Boxplots of R2-score Values for Different Models')
plt.xlabel('Model')
plt.ylabel('R2-score')
# Create custom legend
legend_elements = [Patch(facecolor=palette[i], label=model_names[i]) for i in range(len(model_names))]
plt.legend(handles=legend_elements, title='Models', loc='lower right')
# Save the figure as a PNG file
plt.savefig('boxplot_r2score_24h.png', format='png', dpi=300, bbox_inches='tight')
# Show the plot
plt.show()
