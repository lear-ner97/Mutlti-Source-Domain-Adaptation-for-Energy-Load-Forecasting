# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 14:10:47 2026

@author: umroot
"""


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from functions import *
import time
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from matplotlib.colors import ListedColormap
import random
#import shap


# 0.prepare the dataframes for the analysis



#target building is fixed: Robin_education_Billi
tgt_building = 'Robin_education_Billi'
# choose the source building: 
# 'Robin_education_Derick','Robin_education_Julius',
#                                'Robin_education_Lizbeth','Robin_education_Jasper',
#                                'Robin_education_Terrance','Robin_education_Takako',
#                                'Robin_education_Kristopher','Robin_education_Billi'

#the selected buildings Robin education Julius, Robin education Terrance, Robin education Takako, and 
#Robin education Kristopher are referred to as S1, S2, S3, and S4, respectively.

src_building1 = 'Robin_education_Julius'#Julius S1
src_building2 = 'Robin_education_Terrance'#Terrance S2
src_building3 = 'Robin_education_Takako'#Takako S3
src_building4 = 'Robin_education_Kristopher'#Kristopher S4

# Define the cutoff date
#cutoff_date_src = pd.to_datetime('2017-01-31')
start_date_tgt=pd.to_datetime('2017-01-01')
end_date_tgt = pd.to_datetime('2017-03-01')


# Filter the DataFrame to include only rows until the cutoff date
#filtered_df = df[df['timestamp'] <= cutoff_date]

# upload weather data
# cloud coverage and precipdepth1&6hr were eliminated because they are empty (NAN)
weather_columns = ['timestamp', 'airTemperature', 'windSpeed']
weather_data = pd.read_csv('robin_weather_calendar.csv', parse_dates=[
                           'timestamp'])[weather_columns]
#weather_data=weather_data[weather_data['timestamp'] <= cutoff_date_src]

# 'Robin_education_Derick','Robin_education_Julius',
#                                'Robin_education_Lizbeth','Robin_education_Jasper',
#                                'Robin_education_Terrance','Robin_education_Takako',
#                                'Robin_education_Kristopher','Robin_education_Billi'


src_data1 = pd.read_csv('clean_genome_meters.csv', sep=',')[
    ['timestamp', src_building1]]
src_data1['timestamp'] = pd.to_datetime(src_data1['timestamp'])
src_data1 = pd.merge(src_data1, weather_data, on='timestamp', how='left')
src_data1 = src_data1[src_data1['timestamp'] < start_date_tgt]#data selection only
src_data1.dropna(inplace=True)#remove 22 missing airtemperature values in each dataset

src_data2 = pd.read_csv('clean_genome_meters.csv', sep=',')[
    ['timestamp', src_building2]]
src_data2['timestamp'] = pd.to_datetime(src_data2['timestamp'])
src_data2 = pd.merge(src_data2, weather_data, on='timestamp', how='left')
src_data2 = src_data2[src_data2['timestamp'] < start_date_tgt]
src_data2.dropna(inplace=True)

src_data3 = pd.read_csv('clean_genome_meters.csv', sep=',')[
    ['timestamp', src_building3]]
src_data3['timestamp'] = pd.to_datetime(src_data3['timestamp'])
src_data3 = pd.merge(src_data3, weather_data, on='timestamp', how='left')
src_data3 = src_data3[src_data3['timestamp'] < start_date_tgt]
src_data3.dropna(inplace=True)

src_data4 = pd.read_csv('clean_genome_meters.csv', sep=',')[
    ['timestamp', src_building4]]
src_data4['timestamp'] = pd.to_datetime(src_data4['timestamp'])
src_data4 = pd.merge(src_data4, weather_data, on='timestamp', how='left')
src_data4 = src_data4[src_data4['timestamp'] < start_date_tgt]
src_data4.dropna(inplace=True)

# upload target data
tgt_data = pd.read_csv('clean_genome_meters.csv', sep=',')[
    ['timestamp', tgt_building]]
tgt_data['timestamp'] = pd.to_datetime(tgt_data['timestamp'])
tgt_data=tgt_data[(tgt_data['timestamp'] >= start_date_tgt)&(tgt_data['timestamp'] <= end_date_tgt)]
tgt_data = pd.merge(tgt_data, weather_data, on='timestamp', how='left')
tgt_data.dropna(inplace=True)


dfs = [src_data1,src_data2,src_data3,src_data4,tgt_data]






### 2-weekday vs weekend load comparison between buildings
processed = []

for df in dfs:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.weekday  # 0=Monday, 6=Sunday
    df['is_weekend'] = df['weekday'] >= 5  # True for Saturday & Sunday
    
    # Detect the load column name
    building_name = df.columns[1]
    
    # Compute average load by hour for weekdays
    weekday_avg = df[~df['is_weekend']].groupby('hour')[building_name].mean()
    weekend_avg = df[df['is_weekend']].groupby('hour')[building_name].mean()
    
    # Normalize each curve
    weekday_norm = weekday_avg / weekday_avg.max()
    weekend_norm = weekend_avg / weekend_avg.max()
    
    # Store as DataFrame
    temp_df = pd.DataFrame({
        'hour': list(range(24)),
        'weekday_load': weekday_norm.reindex(range(24), fill_value=0),  # fill missing hours
        'weekend_load': weekend_norm.reindex(range(24), fill_value=0),
        'building': building_name
    })
    processed.append(temp_df)

# Combine all buildings
plot_df = pd.concat(processed)

i=1
# Plot Weekday Curves
plt.figure(figsize=(12,6))
for building in plot_df['building'].unique():
    subset = plot_df[plot_df['building'] == building]
    if i<=4:
        plt.plot(subset['hour'], subset['weekday_load'], label='S'+str(i))
    else:
        plt.plot(subset['hour'], subset['weekday_load'], label='Target')
    i+=1
plt.xlabel('Hour of day')
plt.ylabel('Normalized load')
#plt.title('Normalized Average Weekday Load Curves')
plt.xticks(range(0,24))
plt.grid(True)
plt.legend()
plt.savefig('weekday_load_curves.pdf', format='pdf', bbox_inches='tight')
plt.show()

i=1
# Plot Weekend Curves
plt.figure(figsize=(12,6))
for building in plot_df['building'].unique():
    subset = plot_df[plot_df['building'] == building]
    if i<=4:
        plt.plot(subset['hour'], subset['weekend_load'], label='S'+str(i))
    else:
        plt.plot(subset['hour'], subset['weekend_load'], label='Target')
    i+=1
plt.xlabel('Hour of day')
plt.ylabel('Normalized load')
#plt.title('Normalized Average Weekend Load Curves')
plt.xticks(range(0,24))
plt.grid(True)
plt.legend()
plt.savefig('weekend_load_curves.pdf', format='pdf', bbox_inches='tight')
plt.show()



#3-correlation temperature-load
#a.scatter plot
import matplotlib.pyplot as plt

num_buildings = len(dfs)
cols = 2  # number of columns for subplot layout

# Calculate rows, making sure the last row has a subplot spanning both columns
rows = (num_buildings // cols) + (1 if num_buildings % cols != 0 else 0)

# Create the figure
plt.figure(figsize=(12, 5 * rows))  # adjust height for the number of rows
i=1
for i, df in enumerate(dfs, 1):
    building_name = df.columns[1]  # the load column
    
    # For the last subplot, span both columns
    if i == num_buildings:
        # Use a single index to span both columns
        row_idx = rows  # last row
        col_idx = 1  # span across both columns (set column 1)
    else:
        row_idx = (i - 1) // cols + 1  # calculate row based on index
        col_idx = (i - 1) % cols + 1  # calculate column based on index
    
    # Create subplot with manual positioning
    if i == num_buildings:
        # Last subplot should span two columns (both left and right)
        ax = plt.subplot2grid((rows, cols), (row_idx - 1, 0), colspan=2)
    else:
        ax = plt.subplot(rows, cols, (row_idx - 1) * cols + col_idx)
    
    if i<=4:
        ax.scatter(df['airTemperature'], df[building_name], alpha=0.5, color='steelblue')
        ax.set_xlabel('Air temperature (°C)')
        ax.set_ylabel('Load')
        ax.set_title('S'+str(i))
        ax.grid(True)
    else:
        ax.scatter(df['airTemperature'], df[building_name], alpha=0.5, color='steelblue')
        ax.set_xlabel('Air temperature (°C)')
        ax.set_ylabel('Load')
        ax.set_title('Target')
        ax.grid(True)       

# Adjust layout so the last subplot doesn't overlap
plt.tight_layout()
plt.savefig('scatter_plot.pdf', format='pdf', bbox_inches='tight')
plt.show()



#b. compute linear correlation
correlations = {}

for df in dfs:
    building_name = df.columns[1]
    corr = df[building_name].corr(df['airTemperature'])
    correlations[building_name] = corr

# Print results
for building, corr in correlations.items():
    print(f'{building}: correlation with air temperature = {corr:.2f}')
    


#c. compute non-linear correlation (spearman)
from scipy.stats import spearmanr

spearman_results = {}

for df in dfs:
    building_name = df.columns[1]
    
    # Drop NaNs just in case
    clean_df = df[['airTemperature', building_name]].dropna()
    
    corr, pval = spearmanr(clean_df['airTemperature'], clean_df[building_name])
    
    spearman_results[building_name] = (corr, pval)

# Print results
for building, (corr, pval) in spearman_results.items():
    print(f"{building}: Spearman correlation = {corr:.3f}, p-value = {pval:.3e}")


##d. cool vs hot correlation
import pandas as pd
import numpy as np

processed_dfs = []

for df in dfs:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date #eg 2017-01-01
    df['hour'] = df['timestamp'].dt.hour #eg 23 refers to 11pm
    
    building_name = df.columns[1] #load column
    
    # Compute daily mean temperature
    daily_temp = df.groupby('date')['airTemperature'].mean() # daily mean temperature for each date
    
    # Define thresholds 
    cool_threshold = daily_temp.quantile(0.25) #1st quartile
    hot_threshold = daily_temp.quantile(0.75)  #3rd quartile
    
    # Map classification back to hourly data
    df = df.merge(daily_temp.rename('daily_temp'), on='date') #create daily_temp column, so each timestamp has its corresponding
    #daily mean temperature
    
    df['season_type'] = np.where(df['daily_temp'] <= cool_threshold, 'Cool',
                          np.where(df['daily_temp'] >= hot_threshold, 'Hot', 'Mild')) #mild is assigned if daily_temp is
    #between cool_threshold and hot_threshold
    
    processed_dfs.append(df)


    

from scipy.stats import pearsonr

for df in processed_dfs:
    building_name = df.columns[1] #load column
    
    cool_df = df[df['season_type'] == 'Cool']
    hot_df = df[df['season_type'] == 'Hot']
    
    cool_corr, _ = pearsonr(cool_df['airTemperature'], cool_df[building_name])
    hot_corr, _ = pearsonr(hot_df['airTemperature'], hot_df[building_name])
    
    print(f"{building_name}:")
    print(f"   Correlation in Cool Days = {cool_corr:.3f}")
    print(f"   Correlation in Hot Days  = {hot_corr:.3f}")

