import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# Sample data: Replace this with your actual RMSE values
model_names = ['T', '1S', '2S', '3S', '4S']
rmse_values = {
    'T': [4.189383505182509, 5.503201949848184, 5.06932450989957, 4.941880209066624, 4.460197292325324, 5.1596087560400745, 5.617200978668185, 4.256160759699864, 4.758145737737901, 6.515333043555062],
    '1S': [3.32527689970642, 3.108367011667851, 3.2763901253352823, 3.209427774248051, 3.355955429350682, 3.341409992678829, 3.362077660605619, 3.236440249651763, 3.0694046729840307, 3.326975092860131],
    '2S': [3.350413354671231, 3.1577033507326835, 3.2791279059103164, 3.248761194276065, 3.2102404400224267, 3.182601155767973, 3.3219328394474004, 3.441076992269733, 2.978210312467504, 3.335476468457239],
    '3S': [3.160238240181533, 2.847518958162551, 3.1218373437054754, 3.046351859540346, 3.010121429406233, 3.2565813234874077, 3.1348126267911476, 3.2008816105136133, 2.9948844307764544, 3.169918769274625],
    '4S':[3.3251814930953207, 3.147845559134924, 3.1584485584024202, 3.1838315814602307, 3.3430132098333636, 3.100095789385561, 3.417899665234777, 3.268446142974313, 3.1042929145162583, 3.5356804339456342],
}

# Convert the dictionary to a format suitable for Seaborn
data = []
for model, values in rmse_values.items():
    for value in values:
        data.append((model, value))

# Create a DataFrame from the data
df = pd.DataFrame(data, columns=['Model', 'RMSE'])

# Set up the boxplot with a custom color palette
plt.figure(figsize=(8, 6))
palette = sns.color_palette("husl", len(model_names))  # Choose a color palette
sns.boxplot(x='Model', y='RMSE', data=df, palette=palette)
plt.title('Boxplots of RMSE Values for Different Models')
plt.xlabel('Model')
plt.ylabel('RMSE')
# Create custom legend
legend_elements = [Patch(facecolor=palette[i], label=model_names[i]) for i in range(len(model_names))]
plt.legend(handles=legend_elements, title='Models', loc='upper right')
# Save the figure as a PNG file
plt.savefig('boxplot_rmse_1h.png', format='png', dpi=300, bbox_inches='tight')
# Show the plot
plt.show()
