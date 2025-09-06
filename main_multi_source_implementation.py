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
import numpy as np
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

src_building1 = 'Robin_education_Julius'#Julius
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


src_data1 = pd.read_csv('clean_genome_meters.csv', sep=',')[
    ['timestamp', src_building1]]
src_data1['timestamp'] = pd.to_datetime(src_data1['timestamp'])
src_data1 = pd.merge(src_data1, weather_data, on='timestamp', how='left')
src_data1 = src_data1[src_data1['timestamp'] < start_date_tgt]#data selection only

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


# set the feature matrix in each domain
X_src1 = src_shifted_df1_as_np[:, :-src_horizon]
X_src2 = src_shifted_df2_as_np[:, :-src_horizon]
X_src3 = src_shifted_df3_as_np[:, :-src_horizon]
X_src4 = src_shifted_df4_as_np[:, :-src_horizon]
X_tgt = tgt_shifted_df_as_np[:, :-tgt_horizon]



# set the target
y_src1 = src_shifted_df1_as_np[:, -src_horizon:]
y_src2 = src_shifted_df2_as_np[:, -src_horizon:]
y_src3 = src_shifted_df3_as_np[:, -src_horizon:]
y_src4 = src_shifted_df4_as_np[:, -src_horizon:]

y_tgt = tgt_shifted_df_as_np[:, -tgt_horizon:]



# we use the source data only in training, we don't need it in validation and test
# a full year ends at index 8783 for the source data

X_train_src1,y_train_src1 = X_src1,y_src1#[:8784]
X_train_src2,y_train_src2 = X_src2,y_src2
X_train_src3,y_train_src3 = X_src3,y_src3
X_train_src4,y_train_src4 = X_src4,y_src4

######ONLY for source data selection
#X_valid_src1 = X_src1[7477:]
#y_valid_src1 = y_src1[7477:]

#for 24h horizon: 525,750
#for 1h resolution: 679,1018

#source data selection only
train_limit=525
valid_limit=750
X_train_tgt,y_train_tgt = X_tgt[:train_limit],y_tgt[:train_limit]
X_valid_tgt,y_valid_tgt = X_tgt[train_limit:valid_limit],y_tgt[train_limit:valid_limit]
X_test_tgt,y_test_tgt = X_tgt[valid_limit:],y_tgt[valid_limit:]


# prepare our torch tensors

X_train_src1 = torch.tensor(X_train_src1, dtype=torch.float32)
y_train_src1 = torch.tensor(y_train_src1, dtype=torch.float32)
X_train_src2 = torch.tensor(X_train_src2, dtype=torch.float32)
y_train_src2 = torch.tensor(y_train_src2, dtype=torch.float32)
X_train_src3 = torch.tensor(X_train_src3, dtype=torch.float32)
y_train_src3 = torch.tensor(y_train_src3, dtype=torch.float32)
X_train_src4 = torch.tensor(X_train_src4, dtype=torch.float32)
y_train_src4 = torch.tensor(y_train_src4, dtype=torch.float32)
####uncomment for source data selection
#y_train_tgt = torch.tensor(y_train_tgt, dtype=torch.float32)
#X_valid_src1 = torch.tensor(X_valid_src1, dtype=torch.float32)
#y_valid_src1 = torch.tensor(y_valid_src1, dtype=torch.float32)


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

####uncomment for source data selection
#X_valid_src1 = torch.unsqueeze(X_valid_src1, 2)

X_train_tgt = torch.unsqueeze(X_train_tgt, 2)
X_valid_tgt = torch.unsqueeze(X_valid_tgt, 2)
X_test_tgt = torch.unsqueeze(X_test_tgt, 2)


# Create Custom Datasets
src_train_dataset1 = TimeSeriesDataset(X_train_src1, y_train_src1)
src_train_dataset2 = TimeSeriesDataset(X_train_src2, y_train_src2)
src_train_dataset3 = TimeSeriesDataset(X_train_src3, y_train_src3)
src_train_dataset4 = TimeSeriesDataset(X_train_src4, y_train_src4)

####uncomment for source data selection
#src_valid_dataset1 = TimeSeriesDataset(X_valid_src1, y_valid_src1)

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

####uncomment for source data selection
#src_valid_loader1 = DataLoader(src_valid_dataset1, batch_size=src_batch_size, shuffle=True,pin_memory=True)#,num_workers=2

    
tgt_train_loader = DataLoader(tgt_train_dataset, batch_size=tgt_batch_size, shuffle=False,pin_memory=True)#,num_workers=1
tgt_valid_loader = DataLoader(tgt_valid_dataset, batch_size=tgt_batch_size, shuffle=False,pin_memory=True)
tgt_test_loader = DataLoader(tgt_test_dataset, batch_size=tgt_batch_size, shuffle=False,pin_memory=True)

# print the shape of a batch of data
device = "cuda" if torch.cuda.is_available() else 'cpu' ###change, also in functions.py


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
out_channels1=128

###########bigru hyperparameters
input_size=1  #number of input channels
#hidden_size must be the same as out_channels1, must match the input szie of the frcst layer
hidden_size=128
#tested {100,150,200}
num_gru_units=100 #100 for 24h & 1h
num_gru_layers=1#2 for  24h , 1 for 1h
#fixed hyperparameters
in_channels0=1
forecast_length=24################################ 1 for 1h & 24 for 24h forecasting
output_size=173 # number of features, 173 for 24h, 29 for 1h


seeds=[76]#32, 70, 25, 53, 78, 86, 56, 30, 20, 76]
i=0

for random_seed in seeds:
    i+=1
    print(i)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    full_model=full_model(kernel_size,stride,in_channels0,out_channels0,in_channels1,out_channels1,input_size, num_gru_units, hidden_size, num_gru_layers,hidden_size,hidden_size,forecast_length).to(device)
    
    # define the loss functions
    regression_loss_function = nn.MSELoss()
 
    regression_optimizer = torch.optim.Adam(list(full_model.parameters()), lr=learning_rate)
    
    # schedule the learning rate
    scheduler = lr_scheduler.LinearLR(regression_optimizer, start_factor=1.0, end_factor=0.5, 
                          total_iters=num_epochs)
    
    # initialize the training loss, validation loss and lowest validation loss
    training_loss, validation_loss = [], []
    best_val_loss = float('inf')
    #full_model_path='best_full_model.pth' ####change

    for epoch in range(num_epochs):#src1_features,src2_features,src3_features,src4_features  

        #uncomment this for multi source
        #@for the source features, it depends on how many sources you have. The following line corresponds to four sources
        train_loss,src1_features,src2_features,src3_features,src4_features,target_features=train_one_epoch(full_model,list_source_loaders,tgt_train_loader,
                    regression_loss_function,regression_optimizer,scheduler,epoch,
                    num_epochs)
        #uncomment this for target only
#        train_loss,target_features=train_one_epoch_target_only(full_model,list_source_loaders,src_train_loader1,
#                    regression_loss_function,regression_optimizer,scheduler,epoch,
#                    num_epochs)

        
        
        #this line is always active
        training_loss.append(train_loss)

        val_loss = validate_one_epoch(full_model,epoch,tgt_valid_loader,regression_loss_function)
        validation_loss.append(val_loss)
        
        # Update the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            #change
            torch.save(full_model.state_dict(), best_full_model) 

    end_time = time.time()
    training_time = end_time - start_time
    print(str(training_time)+' s')
    
    # load the best model for the evaluation step
    with torch.no_grad():

        full_model_state_dict = torch.load(best_full_model)

        full_model.load_state_dict(full_model_state_dict)

        predicted = (full_model(X_test_tgt.to(device))).cpu().numpy()
        
    test_predictions = predicted
    
    dummies = np.zeros((X_test_tgt.shape[0], src_shifted_df1_as_np.shape[1]))
    dummies[:, -tgt_horizon:] = test_predictions
    dummies = scaler.inverse_transform(dummies)
    
    #test predictions
    test_predictions = dummies[:, -tgt_horizon:]
    
    dummies = np.zeros((X_test_tgt.shape[0], src_shifted_df1_as_np.shape[1]))
    dummies[:, -tgt_horizon:] = y_test_tgt  
    dummies = scaler.inverse_transform(dummies)
    
    # actual test data
    new_y_test = dummies[:, -tgt_horizon:]
    
        
        
    test_time_range=len(new_y_test[::tgt_horizon, :].flatten())
    test_time[-test_time_range:].to_csv('test_time_24h.csv')
    plt.figure(6)
    plt.plot(test_time[-test_time_range:][:tgt_horizon*7*24], new_y_test[::tgt_horizon, :].flatten()[:tgt_horizon*7*24], label='True', linestyle='--',color='black')#[-len(new_y_test):]
    plt.plot(test_time[-test_time_range:][:tgt_horizon*7*24], test_predictions[::tgt_horizon, :].flatten()[:tgt_horizon*7*24], label='model',color='orange')  # modif_tf[:tgt_horizon*7]

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


    print()
    print("test predictions: \n")
    print("rmse: ", evaluation(test_predictions, new_y_test)[0])
    print("mape: ", evaluation(test_predictions, new_y_test)[1])
    print("r2-score: ", evaluation(test_predictions, new_y_test)[2])


print('rmses:\n')
print(rmses)
print('mapes:\n')
print(mapes)
print('r2scores:\n')
print(r2scores)


