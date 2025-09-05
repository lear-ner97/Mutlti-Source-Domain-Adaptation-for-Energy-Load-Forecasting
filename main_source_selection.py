# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 18:30:10 2024

@author: umroot
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from functions import *
import time
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.manifold import TSNE
import numpy as np
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import random
#import shap






#target building is fixed: Robin_education_Billi
tgt_building = 'Robin_education_Billi'
# choose the source building: 
# 'Robin_education_Derick','Robin_education_Julius',
#                                'Robin_education_Lizbeth','Robin_education_Jasper',
#                                'Robin_education_Terrance','Robin_education_Takako',
#                                'Robin_education_Kristopher','Robin_education_Billi'
nbr_sources=1
src_building1 = 'Robin_education_Derick'#Julius
src_building2 = 'Robin_education_Terrance'#Terrance
src_building3 = 'Robin_education_Takako'#Takako
src_building4 = 'Robin_education_Kristopher'#Kristopher

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

#building_name='Robin_education_Julius'
#data_to_visualize = pd.read_csv('clean_genome_meters.csv', sep=',')[
#    ['timestamp', building_name]]
#data_to_visualize['timestamp'] = pd.to_datetime(data_to_visualize['timestamp'])
#plt.plot(data_to_visualize['timestamp'],data_to_visualize[building_name])
#upload source data
src_data1 = pd.read_csv('clean_genome_meters.csv', sep=',')[
    ['timestamp', src_building1]]
src_data1['timestamp'] = pd.to_datetime(src_data1['timestamp'])
src_data1 = pd.merge(src_data1, weather_data, on='timestamp', how='left')
src_data1 = src_data1[src_data1['timestamp'] < end_date_tgt]#data selection only

src_data2 = pd.read_csv('clean_genome_meters.csv', sep=',')[
    ['timestamp', src_building2]]
src_data2['timestamp'] = pd.to_datetime(src_data2['timestamp'])
src_data2 = pd.merge(src_data2, weather_data, on='timestamp', how='left')
src_data2 = src_data2[src_data2['timestamp'] < start_date_tgt]

src_data3 = pd.read_csv('clean_genome_meters.csv', sep=',')[
    ['timestamp', src_building3]]
src_data3['timestamp'] = pd.to_datetime(src_data3['timestamp'])
src_data3 = pd.merge(src_data3, weather_data, on='timestamp', how='left')
src_data3 = src_data3[src_data3['timestamp'] < start_date_tgt]

src_data4 = pd.read_csv('clean_genome_meters.csv', sep=',')[
    ['timestamp', src_building4]]
src_data4['timestamp'] = pd.to_datetime(src_data4['timestamp'])
src_data4 = pd.merge(src_data4, weather_data, on='timestamp', how='left')
src_data4 = src_data4[src_data4['timestamp'] < start_date_tgt]

# upload target data
tgt_data = pd.read_csv('clean_genome_meters.csv', sep=',')[
    ['timestamp', tgt_building]]
tgt_data['timestamp'] = pd.to_datetime(tgt_data['timestamp'])
tgt_data=tgt_data[(tgt_data['timestamp'] >= start_date_tgt)&(tgt_data['timestamp'] <= end_date_tgt)]
tgt_data = pd.merge(tgt_data, weather_data, on='timestamp', how='left')




# set the historical length T and the future horizon H
src_lookback = 24*7 #24*7
src_horizon = 24 #24*1
tgt_lookback = 24*7
tgt_horizon = 24




# prepare the features: the electricity load lags + weather lags + current weather + current calendar
src_shifted_df1 = prepare_dataframe_for_lstm(src_data1, src_building1, src_lookback, src_horizon)
src_shifted_df2 = prepare_dataframe_for_lstm(src_data2, src_building2, src_lookback, src_horizon)
src_shifted_df3 = prepare_dataframe_for_lstm(src_data3, src_building3, src_lookback, src_horizon)
src_shifted_df4 = prepare_dataframe_for_lstm(src_data4, src_building4, src_lookback, src_horizon)
tgt_shifted_df = prepare_dataframe_for_lstm(tgt_data, tgt_building, tgt_lookback, tgt_horizon)  
calendar_columns = ['timestamp', 'day_of_week', 'cosine_transform_day_of_year', 'cosine_transform_month',
                    'cosine_transform_hour_of_day', 'is_weekend', 'is_holiday']
calendar_data = pd.read_csv('robin_weather_calendar.csv', parse_dates=[
                            'timestamp'])[calendar_columns]



# merge variables and drop target & timestamp
# .drop(['timestamp',src_building], axis=1)
src_shifted_df1 = pd.merge(src_shifted_df1, calendar_data,on='timestamp', how='left')
src_shifted_df2 = pd.merge(src_shifted_df2, calendar_data,on='timestamp', how='left')
src_shifted_df3 = pd.merge(src_shifted_df3, calendar_data,on='timestamp', how='left')
src_shifted_df4 = pd.merge(src_shifted_df4, calendar_data,on='timestamp', how='left')
# drop(['timestamp',tgt_building], axis=1)
tgt_shifted_df = pd.merge(tgt_shifted_df, calendar_data,on='timestamp', how='left')



src_columns1 = ['cosine_transform_hour_of_day', 'is_holiday', 'is_weekend', 'airTemperature', 'windSpeed'] + \
    [f"{src_building1}_lag{i}" for i in range(src_lookback+src_horizon-1, 0, -1)]+[src_building1]
src_columns2 = ['cosine_transform_hour_of_day', 'is_holiday', 'is_weekend', 'airTemperature', 'windSpeed'] + \
    [f"{src_building2}_lag{i}" for i in range(src_lookback+src_horizon-1, 0, -1)]+[src_building2]
src_columns3 = ['cosine_transform_hour_of_day', 'is_holiday', 'is_weekend', 'airTemperature', 'windSpeed'] + \
   [f"{src_building3}_lag{i}" for i in range(src_lookback+src_horizon-1, 0, -1)]+[src_building3]
src_columns4 = ['cosine_transform_hour_of_day', 'is_holiday', 'is_weekend', 'airTemperature', 'windSpeed'] + \
    [f"{src_building4}_lag{i}" for i in range(src_lookback+src_horizon-1, 0, -1)]+[src_building4]
tgt_columns = ['cosine_transform_hour_of_day', 'is_holiday', 'is_weekend', 'airTemperature', 'windSpeed'] + \
    [f"{tgt_building}_lag{i}" for i in range(tgt_lookback+tgt_horizon-1, 0, -1)]+[tgt_building]

#src_columns1 = ['cosine_transform_hour_of_day', 'is_weekend', 'airTemperature'] + \
#    [f"{src_building1}_lag{i}" for i in range(src_lookback+src_horizon-1, 150, -1)]+[src_building1]
#src_columns2 = ['cosine_transform_hour_of_day', 'is_weekend', 'airTemperature'] + \
#    [f"{src_building2}_lag{i}" for i in range(src_lookback+src_horizon-1, 150, -1)]+[src_building2]
#src_columns3 = ['cosine_transform_hour_of_day', 'is_weekend', 'airTemperature'] + \
#    [f"{src_building3}_lag{i}" for i in range(src_lookback+src_horizon-1, 150, -1)]+[src_building3]
#src_columns4 = ['cosine_transform_hour_of_day', 'is_weekend', 'airTemperature'] + \
#    [f"{src_building4}_lag{i}" for i in range(src_lookback+src_horizon-1, 150, -1)]+[src_building4]
#tgt_columns = ['cosine_transform_hour_of_day', 'is_weekend', 'airTemperature'] + \
#    [f"{tgt_building}_lag{i}" for i in range(tgt_lookback+tgt_horizon-1, 150, -1)]+[tgt_building]
    
src_shifted_df1 = src_shifted_df1[src_columns1]
src_shifted_df2 = src_shifted_df2[src_columns2]
src_shifted_df3 = src_shifted_df3[src_columns3]
src_shifted_df4 = src_shifted_df4[src_columns4]
tgt_shifted_df = tgt_shifted_df[tgt_columns]

src_shifted_df1_as_np = src_shifted_df1.to_numpy()
src_shifted_df2_as_np = src_shifted_df2.to_numpy()
src_shifted_df3_as_np = src_shifted_df3.to_numpy()
src_shifted_df4_as_np = src_shifted_df4.to_numpy()
tgt_shifted_df_as_np = tgt_shifted_df.to_numpy()




# this will be used when ploting the curves
period = tgt_data['timestamp']
test_time = period#[-int(len(period)*0.25):]



# data normalization
scaler = StandardScaler()  # MinMaxScaler(feature_range=(-1, 1))##
src_shifted_df1_as_np = scaler.fit_transform(src_shifted_df1_as_np)
src_shifted_df2_as_np = scaler.fit_transform(src_shifted_df2_as_np)
src_shifted_df3_as_np = scaler.fit_transform(src_shifted_df3_as_np)
src_shifted_df4_as_np = scaler.fit_transform(src_shifted_df4_as_np)
tgt_shifted_df_as_np = scaler.fit_transform(tgt_shifted_df_as_np)
#to choose the source datasets
#X_src = scaler.fit_transform(X_src)
# tgt_shifted_df_as_np = scaler.fit_transform(tgt_shifted_df_as_np)

# set the feature matrix in each domain
X_src1 = src_shifted_df1_as_np[:, :-src_horizon]
X_src2 = src_shifted_df2_as_np[:, :-src_horizon]
X_src3 = src_shifted_df3_as_np[:, :-src_horizon]
X_src4 = src_shifted_df4_as_np[:, :-src_horizon]
X_tgt = tgt_shifted_df_as_np[:, :-tgt_horizon]

# already put in the right order: calendar --> weather & load in increasing time order
#X_src=dc(np.flip(X_src, axis=1))
#X_tgt=dc(np.flip(X_tgt, axis=1))

# set the target
y_src1 = src_shifted_df1_as_np[:, -src_horizon:]
y_src2 = src_shifted_df2_as_np[:, -src_horizon:]
y_src3 = src_shifted_df3_as_np[:, -src_horizon:]
y_src4 = src_shifted_df4_as_np[:, -src_horizon:]
#y_src=dc(np.flip(y_src, axis=1))
y_tgt = tgt_shifted_df_as_np[:, -tgt_horizon:]
#y_tgt=dc(np.flip(y_tgt, axis=1))


# we use the source data only in training, we don't need it in validation and test
# a full year ends at index 8783 for the source data
######ONLY for source data selection
X_train_src1,y_train_src1 = X_src1[:7477],y_src1[:7477]#[:8784]
X_train_src2,y_train_src2 = X_src2,y_src2
X_train_src3,y_train_src3 = X_src3,y_src3
X_train_src4,y_train_src4 = X_src4,y_src4

X_valid_src1 = X_src1[7477:]
y_valid_src1 = y_src1[7477:]
#train_range = 0.6
#valid_range = train_range+0.2
# train-validation-test split
# 50%-25%-25% train-valid-test
#for 24h horizon: 525,750
#for 1h resolution: 679,1018

#source data selection only
train_limit=525
valid_limit=750
X_train_tgt,y_train_tgt = X_tgt[:train_limit],y_tgt[:train_limit]
X_valid_tgt,y_valid_tgt = X_tgt[train_limit:valid_limit],y_tgt[train_limit:valid_limit]
X_test_tgt,y_test_tgt = X_tgt,y_tgt#[valid_limit:],y_tgt#[valid_limit:]
#X_valid_tgt = X_tgt[int(len(X_tgt)*train_range):int(len(X_tgt)*valid_range)]
#y_valid_tgt = y_tgt[int(len(X_tgt)*train_range):int(len(X_tgt)*valid_range)]
#we will test the model on the full target data
#X_test_tgt = X_tgt#[int(len(X_tgt)*valid_range):]
#y_test_tgt = y_tgt#[int(len(X_tgt)*valid_range):]


# prepare our torch tensors

X_train_src1 = torch.tensor(X_train_src1, dtype=torch.float32)
y_train_src1 = torch.tensor(y_train_src1, dtype=torch.float32)
X_train_src2 = torch.tensor(X_train_src2, dtype=torch.float32)
y_train_src2 = torch.tensor(y_train_src2, dtype=torch.float32)
X_train_src3 = torch.tensor(X_train_src3, dtype=torch.float32)
y_train_src3 = torch.tensor(y_train_src3, dtype=torch.float32)
X_train_src4 = torch.tensor(X_train_src4, dtype=torch.float32)
y_train_src4 = torch.tensor(y_train_src4, dtype=torch.float32)
#y_train_tgt = torch.tensor(y_train_tgt, dtype=torch.float32)
X_valid_src1 = torch.tensor(X_valid_src1, dtype=torch.float32)
y_valid_src1 = torch.tensor(y_valid_src1, dtype=torch.float32)


X_train_tgt = torch.tensor(X_train_tgt, dtype=torch.float32)
y_train_tgt = torch.tensor(y_train_tgt, dtype=torch.float32)
X_valid_tgt = torch.tensor(X_valid_tgt, dtype=torch.float32)
y_valid_tgt = torch.tensor(y_valid_tgt, dtype=torch.float32)
X_test_tgt = torch.tensor(X_test_tgt, dtype=torch.float32)
y_test_tgt = torch.tensor(y_test_tgt, dtype=torch.float32)

# add a dimension to each tensor
X_train_src1 = torch.unsqueeze(X_train_src1, 2)
X_train_src2 = torch.unsqueeze(X_train_src2, 2)
X_train_src3 = torch.unsqueeze(X_train_src3, 2)
X_train_src4 = torch.unsqueeze(X_train_src4, 2)

X_valid_src1 = torch.unsqueeze(X_valid_src1, 2)
#X_train_tgt = torch.unsqueeze(X_train_tgt, 2)
#X_valid_tgt = torch.unsqueeze(X_valid_tgt, 2)
X_train_tgt = torch.unsqueeze(X_train_tgt, 2)
X_valid_tgt = torch.unsqueeze(X_valid_tgt, 2)
X_test_tgt = torch.unsqueeze(X_test_tgt, 2)


# Create Custom Datasets
src_train_dataset1 = TimeSeriesDataset(X_train_src1, y_train_src1)
src_train_dataset2 = TimeSeriesDataset(X_train_src2, y_train_src2)
src_train_dataset3 = TimeSeriesDataset(X_train_src3, y_train_src3)
src_train_dataset4 = TimeSeriesDataset(X_train_src4, y_train_src4)

src_valid_dataset1 = TimeSeriesDataset(X_valid_src1, y_valid_src1)
#tgt_train_dataset = TimeSeriesDataset(X_train_tgt, y_train_tgt)
#tgt_valid_dataset = TimeSeriesDataset(X_valid_tgt, y_valid_tgt)
tgt_train_dataset = TimeSeriesDataset(X_train_tgt, y_train_tgt)
tgt_valid_dataset = TimeSeriesDataset(X_valid_tgt, y_valid_tgt)
tgt_test_dataset = TimeSeriesDataset(X_test_tgt, y_test_tgt)


src_batch_size = 128
tgt_batch_size = 16


# prepare our data for training with dataloader

src_train_loader1 = DataLoader(src_train_dataset1, batch_size=src_batch_size, shuffle=True,pin_memory=True)#,num_workers=2
src_train_loader2 = DataLoader(src_train_dataset2, batch_size=src_batch_size, shuffle=True,pin_memory=True)
src_train_loader3 = DataLoader(src_train_dataset3, batch_size=src_batch_size, shuffle=True,pin_memory=True)
src_train_loader4 = DataLoader(src_train_dataset4, batch_size=src_batch_size, shuffle=True,pin_memory=True)
list_source_loaders=[src_train_loader1,src_train_loader2,src_train_loader3,src_train_loader4]

src_valid_loader1 = DataLoader(src_valid_dataset1, batch_size=src_batch_size, shuffle=True,pin_memory=True)#,num_workers=2
# for source_data,(tgt_batch_index, tgt_batch) in zip(
#         zip(*list_source_loaders),enumerate(tgt_train_loader)):
#     for sd in source_data:
#         for data,
#         src_x_batch, src_y_batch = sd[0].to(device), sd[1].to(device)
#         print(src_x_batch.shape)
#     break
#for (batch1, batch2, batch3, batch4) in zip(src_train_loader1, src_train_loader2, src_train_loader3, src_train_loader4):
    
    
# l1=[1,3,5,7]
# l2=[2,4,6,8]
# for (i,j) in zip(l1,l2):
#     print(i,j)
# = DataLoader(src_valid_dataset, batch_size=src_batch_size, shuffle=False)
# tgt_train_loader = DataLoader(
#     tgt_train_dataset, batch_size=tgt_batch_size, shuffle=True)
# tgt_valid_loader = DataLoader(
#     tgt_valid_dataset, batch_size=tgt_batch_size, shuffle=False)
tgt_train_loader = DataLoader(tgt_train_dataset, batch_size=tgt_batch_size, shuffle=False,pin_memory=True)#,num_workers=1
tgt_valid_loader = DataLoader(tgt_valid_dataset, batch_size=tgt_batch_size, shuffle=False,pin_memory=True)
tgt_test_loader = DataLoader(tgt_test_dataset, batch_size=tgt_batch_size, shuffle=False,pin_memory=True)

# print the shape of a batch of data
device = "cuda" if torch.cuda.is_available() else 'cpu'

print("source data\n")
i = 0
for _, batch in enumerate(src_train_loader1):
    print(i)
    i += 1
    x_batch_src, y_batch_src = batch[0].to(device), batch[1].to(device)
    print(x_batch_src.shape, y_batch_src.shape)
    break
print("target data\n")
#tgt_train_loader finishes at batch 19
for _, batch in enumerate(tgt_train_loader):
    x_batch_tgt, y_batch_tgt = batch[0].to(device), batch[1].to(device)
    print(x_batch_tgt.shape, y_batch_tgt.shape)
    break



best_TCN_feature_extractor_path = 'best_TCN_feature_extractor.pth'
best_BiGRU_feature_extractor_path = 'best_BiGRU_feature_extractor.pth'
best_Attention_path = 'best_Attention.pth'
best_ForecastingLayer_path = 'best_ForecastingLayer.pth'
best_full_model='best_full_model.pth'
# compute the trainnig time
start_time = time.time()


rmses = []
mapes = []
r2scores = []
i = 0
random_seed=32 # the result is basically the same with different random seeds
models=['modif_tf']#,'cnn_lstm','tf']




# 4-fix the seed16
# for reproducibility of the results of Table 7 set seeds=[700]
# for reproducibility of the results of Tables 8 set seeds=[76, 88, 91, 33, 55, 87, 3, 62, 50, 21]
# the seeds were generated randomly using the next line


#seeds = [700]
#training hyperparameters
learning_rate = 0.001
num_epochs = 20
############### tcn hyperparameters
# kernel_size was tested using these vlaues{3,5,7,9,10,11,13}
# for 3,5 gave low accuracy values, started to improve significantly from 7, the best is 9
kernel_size=  7#7 for 24h, 9 for 1h gave .9671
#stride was fixed to 1 in order to avoid dimensionality matching issues
stride= 1 #could not be tuned because of dimentionality matching in tcn/out_0 = self.relu(x0 + res0)
# out_channels0 {64,128}
out_channels0= 128#128 # better than 64 when both out_s1 and out_s0 are 128, 128 for 24h
in_channels1= 75 # has no role, to be deleted
# out_channels0 {64,128}
out_channels1=128#128 for 24h

###########bigru hyperparameters
input_size=1  #fixed
#hidden_size must be the same as out_channels1, must match the input szie of the frcst layer
hidden_size=128#128 for 24H
#tested {100,150,200}
num_gru_units=100 #100 for 24h & 1h
num_gru_layers=2#2 for  24h , 1 for 1h
#fixed hyperparameters
in_channels0=1
forecast_length=24################################ 1 for 1h & 24 for 24h forecasting
output_size=173 # number of features, 173 for 24h, 29 for 1h

#seeds=random.sample(range(1, 100), 5)
#30 is the random seed used for 24h
#76 is the random seed used for 1h
seeds=[76]#32, 70, 25, 53, 78, 86, 56, 30, 20, 76]
i=0

for random_seed in seeds:
    i+=1
    print(i)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    
    #TCN_feature_extractor=TCN(kernel_size,stride,in_channels0,out_channels0,in_channels1,out_channels1).to(device)
    #BiGRU_feature_extractor=BiGRU(input_size, num_gru_units, hidden_size, num_gru_layers).to(device)
    #Attention_mechanism=Attention(hidden_size).to(device)
    #Forecasting_Layer=ForecastingLayer(hidden_size,forecast_length).to(device)
    #feature_extractor=feature_extractor(kernel_size,stride,in_channels0,out_channels0,in_channels1,out_channels1,input_size, num_gru_units, hidden_size, num_gru_layers,hidden_size)
    full_model=full_model(kernel_size,stride,in_channels0,out_channels0,in_channels1,out_channels1,input_size, num_gru_units, hidden_size, num_gru_layers,hidden_size,hidden_size,forecast_length).to(device)
    
    # define the loss functions
    regression_loss_function = nn.MSELoss()
    #disc_loss_function = nn.CrossEntropyLoss()
    
    # define the optimizers
    #disc_optimizer = torch.optim.Adam(
     #   discriminator.parameters(), lr=learning_rate)
    #regression_optimizer = torch.optim.Adam(list(TCN_feature_extractor.parameters())+list(BiGRU_feature_extractor.parameters())
    #                     + list(Attention_mechanism.parameters())+list(Forecasting_Layer.parameters()), lr=learning_rate)
    regression_optimizer = torch.optim.Adam(list(full_model.parameters()), lr=learning_rate)
    # schedule the learning rate
    scheduler = lr_scheduler.LinearLR(regression_optimizer, start_factor=1.0, end_factor=0.5, 
                          total_iters=num_epochs)
    
    # initialize the training loss, validation loss and lowest validation loss
    training_loss, validation_loss = [], []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):#src1_features,src2_features,src3_features,src4_features  
#        train_loss,src1_features,src2_features,src3_features,target_features=train_one_epoch(TCN_feature_extractor,BiGRU_feature_extractor,
#                            Attention_mechanism,Forecasting_Layer,list_source_loaders,tgt_train_loader,
#                            regression_loss_function,regression_optimizer,scheduler,epoch,
#                            num_epochs)
        #uncomment this for multi source
#        train_loss,src1_features,src2_features,src3_features,target_features=train_one_epoch(full_model,list_source_loaders,tgt_train_loader,
#                    regression_loss_function,regression_optimizer,scheduler,epoch,
#                    num_epochs)
        #uncomment this for target only
        train_loss,target_features=train_one_epoch_target_only(full_model,list_source_loaders,src_train_loader1,
                    regression_loss_function,regression_optimizer,scheduler,epoch,
                    num_epochs)

        
        
        #this line is always active
        training_loss.append(train_loss)
        # if you want to train the TF-LSTM without DA then uncomment the following 3 lines, comment the previous 3 lines and remove the source decoder parameters in the optimizer
        # training_loss.append(train_one_epoch_withoutDA(feature_extractor,tgt_generator,
        #                     tgt_train_loader,gen_loss_function,gen_optimizer,
        #                     scheduler,epoch,num_epochs))
#        val_loss = validate_one_epoch(TCN_feature_extractor,BiGRU_feature_extractor,Attention_mechanism,
#                                      Forecasting_Layer,epoch,tgt_valid_loader,regression_loss_function)
        #normally tgt_valid_loader
        val_loss = validate_one_epoch(full_model,epoch,src_valid_loader1,regression_loss_function)
        validation_loss.append(val_loss)
        # Update the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # save the best model
#            best_TCN_feature_extractor_path = 'best_TCN_feature_extractor.pth'
#            best_BiGRU_feature_extractor_path = 'best_BiGRU_feature_extractor.pth'
#            best_Attention_path = 'best_Attention.pth'
#            best_ForecastingLayer_path = 'best_ForecastingLayer.pth'
#            torch.save(TCN_feature_extractor.state_dict(),
#                       best_TCN_feature_extractor_path)
#            torch.save(BiGRU_feature_extractor.state_dict(), best_BiGRU_feature_extractor_path)
#            torch.save(Attention_mechanism.state_dict(), best_Attention_path)
#            torch.save(Forecasting_Layer.state_dict(), best_ForecastingLayer_path)
            torch.save(full_model.state_dict(), best_full_model)

    end_time = time.time()
    training_time = end_time - start_time
    print(str(training_time)+' s')
    
    # load the best model for the evaluation step
    with torch.no_grad():
#        TCN_feature_extractor_state_dict = torch.load(best_TCN_feature_extractor_path)
#        BiGRU_feature_extractor_state_dict = torch.load(best_BiGRU_feature_extractor_path)
#        Attention_state_dict = torch.load(best_Attention_path)
#        ForecastingLayer_state_dict = torch.load(best_ForecastingLayer_path)
        full_model_state_dict = torch.load(best_full_model)
        
#        TCN_feature_extractor.load_state_dict(TCN_feature_extractor_state_dict)
#        BiGRU_feature_extractor.load_state_dict(BiGRU_feature_extractor_state_dict)
#        Attention_mechanism.load_state_dict(Attention_state_dict)
#        Forecasting_Layer.load_state_dict(ForecastingLayer_state_dict)
        full_model.load_state_dict(full_model_state_dict)
        # uncomment the [0] if your encoder is a transformer
        # comment the [0] if your encoder is a CNN
#        tcn_predicted = TCN_feature_extractor(X_test_tgt.to(device))  # [0]
#        bigru_predicted = BiGRU_feature_extractor(X_test_tgt.to(device))
#        predicted_features=tcn_predicted+bigru_predicted
#        final_features=Attention_mechanism(predicted_features)[0]
#        predicted = (Forecasting_Layer(final_features)).cpu().numpy()
        predicted = (full_model(X_test_tgt.to(device))).cpu().numpy()
        
    test_predictions = predicted
    
    dummies = np.zeros((X_test_tgt.shape[0], src_shifted_df1_as_np.shape[1]))
    dummies[:, -tgt_horizon:] = test_predictions
    dummies = scaler.inverse_transform(dummies)
    
    #test predictions
    test_predictions = dummies[:, -tgt_horizon:]
    
    dummies = np.zeros((X_test_tgt.shape[0], src_shifted_df1_as_np.shape[1]))
    dummies[:, -tgt_horizon:] = y_test_tgt  # [:,:24]
    dummies = scaler.inverse_transform(dummies)
    
    # actual test data
    new_y_test = dummies[:, -tgt_horizon:]
    
        
        
    test_time_range=len(new_y_test[::tgt_horizon, :].flatten())
    test_time[-test_time_range:].to_csv('test_time_24h.csv')
    plt.figure(6)
    plt.plot(test_time[-test_time_range:][:tgt_horizon*7*24], new_y_test[::tgt_horizon, :].flatten()[:tgt_horizon*7*24], label='True', linestyle='--',color='black')#[-len(new_y_test):]
    plt.plot(test_time[-test_time_range:][:tgt_horizon*7*24], test_predictions[::tgt_horizon, :].flatten()[:tgt_horizon*7*24], label='model',color='orange')  # modif_tf[:tgt_horizon*7]
    # plt.plot(test_time[-test_time_range:][:tgt_horizon*7],test_predictions_cnn_lstm[::tgt_horizon,:].flatten()[:tgt_horizon*7], label='CNN-LSTM DAF',color='green')
    # plt.plot(test_time[-test_time_range:][:tgt_horizon*7],test_predictions_tf[::tgt_horizon,:].flatten()[:tgt_horizon*7], label='Traditional TF DAF',color='blue')
    plt.xlabel('time')
    plt.ylabel('load (kWh)')
    plt.gcf().autofmt_xdate() 
    # plt.title('testing')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    #plt.savefig(tgt_building+str(weeks)+'.png', bbox_inches='tight')
    plt.show()
    # #######################################################################
    epochs = [i for i in range(1, num_epochs+1)]
    fig, ax = plt.subplots()
    # , label='Training Loss', color='blue')#, linestyle='-', marker='o')
    ax.plot(epochs, training_loss, label='training loss')
    ax.plot(epochs, validation_loss, label='Validation Loss')#, linestyle='--', marker='x')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    plt.xticks(rotation=45, ha='right')
    # Limit number of x-axis ticks
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend()
    plt.tight_layout()  # Adjust layout to fit labels
    plt.show()
    
    
    rmses.append(evaluation(test_predictions, new_y_test)[0])
    mapes.append(evaluation(test_predictions, new_y_test)[1])
    r2scores.append(evaluation(test_predictions, new_y_test)[2])

# print("training metrics:\n")
# print("rmse: ", evaluation(train_predictions, new_y_train)[0])
# print("mape: ", evaluation(train_predictions, new_y_train)[1])
# print("r2-score: ", evaluation(train_predictions, new_y_train)[2])

    print()
    print("test predictions: \n")
    print("rmse: ", evaluation(test_predictions, new_y_test)[0])
    print("mape: ", evaluation(test_predictions, new_y_test)[1])
    print("r2-score: ", evaluation(test_predictions, new_y_test)[2])
    
    #store the y_true and y_hat in a csv file in order to compute the error distribution and plot prediction vs true
#    results_df = pd.DataFrame({
#    'y_true': new_y_test.reshape(-1),
#    'y_hat': test_predictions.reshape(-1)
#})
#
#    results_df.to_csv('test_results_1S_24h.csv', index=False)
#test_results_1S_24h.csv should be reimplemented
    

print('rmses:\n')
print(rmses)
print('mapes:\n')
print(mapes)
print('r2scores:\n')
print(r2scores)
#why t-sne and not pca?
#because pca works better on linear data, whereas t-sne has no restriction
#dim=5
#src1_features_=src1_features[:target_features.shape[0]]#,:dim
##src2_features_=src2_features[:target_features.shape[0]]
##src3_features_=src3_features[:target_features.shape[0]]
##src4_features_=src4_features[:target_features.shape[0]]
#target_features_=target_features#[:,:dim]
##src1_features=src1_features[:target_features.shape[0],:dim]
##src2_features=src2_features[:target_features.shape[0],:dim]
##src3_features=src3_features[:target_features.shape[0],:dim]
##src4_features=src4_features[:target_features.shape[0],dim]
##target_features=target_features[:,:dim]
## Combine features into a single tensor and create labels
##, ,
#features = torch.cat([src1_features_,target_features])#src2_features_,src3_features_ ,src4_features_,  target_features_], dim=0)
#features_cpu = features.detach().cpu().numpy()  # Transfer to CPU and convert to NumPy array
##features_cpu=np.squeeze(features_cpu)
## Create labels
##
#labels = np.array(#src2_features_.shape[0]
#        [1] * src1_features_.shape[0] + [2] * target_features_.shape[0]#   + [3] * src3_features_.shape[0] + [4] * src4_features_.shape[0] + [5] * target_features_.shape[0]
#        )
#
## Apply PCA for initial dimensionality reduction
#pca = PCA(n_components=2)
#features_pca = pca.fit_transform(features_cpu)
#
#pca.fit(features_cpu)
#pca.explained_variance_ratio_
#
## Assuming labels are integers from 0 to 4 (or 1 to 5 if adjusted)
## Assuming labels are integers from 0 to 4 (or 1 to 5 if adjusted)
#labels = labels.astype(int)
#unique_labels = np.unique(labels)
#
## Create a discrete colormap with only the colors you need
#cmap = ListedColormap(plt.cm.tab10.colors[:len(unique_labels)])
#plt.figure(figsize=(8, 6))
#scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels, cmap=cmap, s=10)
## Create a colorbar with specific ticks and labels
#cbar = plt.colorbar(scatter)
#cbar.set_ticks(unique_labels)  # Set ticks to unique labels
#cbar.set_ticklabels(unique_labels)  # Set tick labels to unique labels
#plt.title('PCA of the source and target features')
#plt.xlabel('Principal Component 1')
#plt.ylabel('Principal Component 2')
#plt.savefig(str(nbr_sources)+'S_pca.png', bbox_inches='tight')
#plt.show()
## Initialize and fit t-SNE
##perplexity must be lower than the #samples
##perplexity is usually between 5 and 50, the higher tthe #samples the higher it is
#tsne = TSNE(n_components=2, random_state=42,perplexity=10, n_iter=250,learning_rate='auto',
#                  init='random')
#start_time = time.time()
#features_tsne = tsne.fit_transform(features_pca)
#end_time = time.time()
#training_time = end_time - start_time
#print('time to train the tsne: ' + str(training_time) + str('s'))
#
## Create a scatter plot
##
#plt.figure(figsize=(8, 6))
#scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=np.array(labels).astype(int), cmap='viridis')
## Add a legend
#plt.legend(*scatter.legend_elements(), title="Datasets")
#plt.title("t-SNE Visualization of Features")
#plt.xlabel("t-SNE Component 1")
#plt.ylabel("t-SNE Component 2")
#plt.xlim(-10, 10)
#ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
## Show the plot
#plt.tight_layout() 
#plt.savefig(str(nbr_sources)+'S_tsne.png', bbox_inches='tight')
#plt.show()

#uncomment the following code to get the attention mechanism explainability
##########################################    
#tcn_predicted = TCN_feature_extractor(X_train_src4.to(device))  # [0]
#bigru_predicted = BiGRU_feature_extractor(X_train_src4.to(device))
#predicted_features=tcn_predicted+bigru_predicted
#attention_weights=Attention_mechanism(predicted_features)[1].squeeze()
#
## Visualize the attention weights as a heatmap
#plt.figure(figsize=(8, 6))
#plt.imshow(attention_weights.cpu().detach().numpy(), cmap='viridis', aspect='auto')
#plt.colorbar(label='Attention Weight')
#plt.title('Attention Weights Heatmap')
#plt.xlabel('Feature Index')
#plt.ylabel('Sample Index')
#plt.savefig('attention_weights_heatmap.png', bbox_inches='tight')
#plt.show()
#
## Average attention weights across all samples (mean along axis 0)
#average_attention = attention_weights.mean(dim=0)
## Plot average attention weights across features
#average_weight_per_feature=average_attention.cpu().detach().numpy()
#plt.bar(range(attention_weights.size(1)), average_weight_per_feature)
#plt.title('Average Attention Weights Across All Samples')
#plt.xlabel('Feature Index')
#plt.ylabel('Average Attention Weight')
#plt.savefig('average_attention_weights.png', bbox_inches='tight')
#plt.show()
#
## Get the indices of the top 3 highest values
#top_n = 10
#top_n_indices = np.argsort(average_weight_per_feature)[-top_n:][::-1]
#top_n_values = average_weight_per_feature[top_n_indices]
#
#print(f"Top {top_n} highest values and their indices:")
#for idx, value in zip(top_n_indices, top_n_values):
#    print(f"Index: {idx}, Value: {value}")
##in average 20(lag 176~24*6+8),21(lag 175~24*6+7) are the most influential features
## 19 (lag 177~24*6+9),18(lag 178~24*6+10),22(lag 174~24*6+6),23(lag 173~24*6+5) come next in the list, but with lower weights
###########################################
##for 1hforecasting, the most important features are:
##lag1 (remarkably high), lag2, lag3
##then comes lags 4,5,6,7,.. with relativvely lower weights
##the same results are observed for tgt_train and tgt_test
#    
## Extract the first 10 samples and columns starting from index 5 (only electricity)
#nbr_samples=5
#randomness=140 #an integer between 0 and 291 (to not go beyond the number of samples)
#samples_to_plot = X_test_tgt[randomness:randomness+nbr_samples, 5:].squeeze()
#future_values = y_test_tgt[randomness:randomness+nbr_samples]  # Corresponding prediction vectors
## Define the range of columns to highlight (18 to 24 in the full dataset)
#highlight_columns = list(range(13, 19))  # 13 to 19 correspond to the indexes of 
##the most important features
#
## Define a color for the highlighted columns
#highlight_color = 'red'
#default_color = 'blue'
#prediction_color = 'green'
## Set up the plot
#plt.figure(figsize=(8, 6))
#
## Plot the first nbr_samples samples
#for i in range(nbr_samples):
#    if i<nbr_samples-1:
#        sample = samples_to_plot[i]
#        prediction = future_values[i]
#        # Plot the entire sample (blue color)
#        plt.plot(sample, color=default_color, alpha=0.5)  # Plot with some transparency
#        
#        # Highlight the important columns (18 to 24) in a different color (red)
#        plt.plot(range(13, 19), sample[13:19], color=highlight_color, linewidth=2)
#    
#        # Plot the prediction vector (green color), which comes after the sample
#        plt.plot(range(len(sample), len(sample) + len(prediction)), prediction, color=prediction_color, linewidth=2)
#    else:
#        sample = samples_to_plot[i]
#        prediction = future_values[i]
#        # Plot the entire sample (blue color)
#        plt.plot(sample, color=default_color, alpha=0.5,label="historical loads")  # Plot with some transparency
#        
#        # Highlight the important columns (18 to 24) in a different color (red)
#        plt.plot(range(13, 19), sample[13:19], color=highlight_color, linewidth=2,label="important lags")
#    
#        # Plot the prediction vector (green color), which comes after the sample
#        plt.plot(range(len(sample), len(sample) + len(prediction)), prediction, color=prediction_color, linewidth=2,label="future loads")
## Add labels and title
#plt.xlabel('time Index')
#plt.ylabel('Normalized load value')
#plt.title('Future 24h Values and Highlighted Lags from the 7th Past Day')
## Add a legend
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
## Show the plot
#plt.savefig('lags_140.png', bbox_inches='tight')
#plt.show()

#weakday
#200: future is complete daily load curve with a peak, highlights are descending half daily curve
#20: future is complete load curve with a trough, highlights are ascending half daily curve
#100: future comlete with peak, highlights are a descnding half
#weekend
#50: future is at the same time with peak & trough, highlights are ascending half
#250: future is a complete load curve with a peak, highlights are a descending curve
#60: highlights are half descending, the future is complete with peak 

#shap explainability
######1-global explainability
##############################comment
#import shap
#from shap import DeepExplainer,KernelExplainer,GradientExplainer
#
#
## You can also use a random subset from the training set to get a balanced representation
#background_data = X_train_tgt.to(device)#[np.random.choice(X_train_tgt.shape[0], size=500, replace=False)].to(device)
##using all the training data as background data for a better fit of the explainer
##to the data distribution
## Initialize the SHAP explainer, here we use gradient explainer
##DeepExplainer gave an error of threshold error
#explainer = GradientExplainer(full_model, background_data)
##the pred function of the model has to return a single output not a tuple
#full_model.train(True) 
##compute the shap values
#
##number of samples to explain
#nbr_samples_to_explain=100
## Get SHAP values for a sample from your test data
## the test data time period is in January and february, so choosing any 100
## random points will be convenient
##explainer.shap_values
#shap_values = explainer.shap_values(X_test_tgt[:nbr_samples_to_explain].to(device))  # X_test is the data you want to explain
##the more data you explain the broader and more generalizing is your explanation
##the smaller the data size the more local is the explanation
##shap.plots.bar(shap_values)
#
#feature_names=['cosine_transform_hour_of_day','is_holiday','is_weekend','airTemperature','windSpeed']+[f"Lag {i}" for i in range(src_lookback,0,-1)]
#
#avg_shap_values = np.mean(np.array(shap_values), axis=0).squeeze(axis=-1) 
#
#shap.summary_plot(avg_shap_values, X_test_tgt[:nbr_samples_to_explain].cpu().numpy().squeeze(), feature_names=feature_names,max_display=50)
##plot the bar plot
#shap.summary_plot(avg_shap_values, X_test_tgt[:nbr_samples_to_explain], feature_names=feature_names, plot_type="bar")#,max_display=12)
##dependance plot
#shap.dependence_plot(0, avg_shap_values, X_test_tgt[:nbr_samples_to_explain].cpu().numpy().squeeze(), feature_names=feature_names,  show=True,interaction_index=3)#,with temperature
##plt.show()
#for horizon_idx in range(7,8):#src_horizon  # Loop through the 24 forecast horizons
#    #comment the next line when H=1
#    shap_vals_at_horizon = shap_values[horizon_idx]  # Shape: (num_samples, num_time_steps, 1)
#    
#    # Squeeze the last dimension to get rid of the single value (it’s not needed for plotting)
#    shap_vals_at_horizon = shap_vals_at_horizon.squeeze(axis=-1)  # Now shape: (num_samples, num_time_steps)
#    #shap_vals_at_horizon in the last line
#    #shap.plots.bar(shap_values)
#    # Create a summary plot for this specific forecast horizon
#    print(f"Summary plot for forecast horizon {horizon_idx + 1}")
#    #plot the summary plot
#    shap.summary_plot(shap_vals_at_horizon, X_test_tgt[:nbr_samples_to_explain].cpu().numpy().squeeze(), feature_names=feature_names)
#    #plot the bar plot
#    shap.summary_plot(shap_vals_at_horizon, X_test_tgt[:nbr_samples_to_explain], feature_names=feature_names, plot_type="bar",max_display=12)
#    #dependance plot
#    shap.dependence_plot(0, shap_vals_at_horizon, X_test_tgt[:nbr_samples_to_explain].cpu().numpy().squeeze(), feature_names=feature_names,  show=True,interaction_index=3)#,with temperature
#    plt.show()
    #['cosine_transform_hour_of_day', 'is_holiday', 'is_weekend', 'airTemperature', 'windSpeed']
#     Plot SHAP interaction plot for two features
#    index 0 is for cosine transform
    ##################################

#gradient clipping to avoid explosive gradient

#Features (time steps) are sorted by importance: By default, the summary plot orders 
#the time steps by the average magnitude of their Shapley values across the samples. 
#This means that the most important time steps (i.e., those contributing the most to 
#the model's prediction) will appear first in the plot.

#The color gradient in the plot will represent the magnitude of the Shapley values, 
#where positive contributions will push the prediction up, and negative ones will 
#push it down.

##2-local interpretation
#import shap
#
## Let's assume you're interested in a specific sample and forecast horizon.
#sample_idx = 0  # Example: selecting the first sample
#horizon_idx = 23  # Example: selecting the 24th forecast step
#
## Get the Shapley values for the selected sample at the given forecast horizon
#shap_vals_at_horizon = shap_values[horizon_idx][sample_idx].squeeze()  # Shape: (num_time_steps, )
#
## Create the force plot for the selected sample and forecast horizon
#shap.initjs()  # Initialize JS for interactive plotting
#shap.force_plot(shap_vals_at_horizon, X_test_tgt[sample_idx].cpu().numpy(), feature_names=feature_names)
