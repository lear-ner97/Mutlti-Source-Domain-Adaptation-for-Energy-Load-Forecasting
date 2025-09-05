# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 22:32:04 2024

@author: umroot
"""
from copy import deepcopy as dc
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset
import torch.nn.functional as F
from sklearn.metrics import r2_score ,mean_squared_error
import numpy as np
import torch.distributed as dist

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def prepare_dataframe_for_lstm(df, building,lookback,horizon):
    df = dc(df)
    df.set_index('timestamp', inplace=True)#time
    for column in df.columns:
        if column == building:  # Exclude timestamp column
            for lag in range(1, lookback + horizon):
                df[f'{column}_lag{lag}'] = df[column].shift(lag)
        else:
            for lag in range(1, lookback + 1):
                df[f'{column}_lag{lag}'] = df[column].shift(lag)
    # Drop rows with NaN values (resulting from shifting)
    df.dropna(inplace=True)
    
    return df



########## TCN #############################
torch.backends.cudnn.benchmark = True  # might be required to fasten TCN


class Chomp1d(nn.Module):
    # to remove extra padding added to maintain the input-output length alignment when using 
    # convolutions with dilations
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TCN(nn.Module):
    def __init__(self,kernel_size,stride,in_channels0,out_channels0,in_channels1,out_channels1):# self, configs/replace configs with fixed values and apply the model on an example data ??????????
        super(TCN, self).__init__()

        dilation0 = 1
        padding0 = (kernel_size - 1) * dilation0

        dilation1 = 2
        padding1 = (kernel_size - 1) * dilation1


        self.relu = nn.LeakyReLU()


        #conv1d by default does not use future context. It only uses past and current context
        #so we can consider our network as causal
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels0, out_channels0, kernel_size=kernel_size, stride=stride, bias=True, padding=padding0,
                      dilation=dilation0),
            Chomp1d(padding0),
            nn.BatchNorm1d(out_channels0),
            nn.LeakyReLU(),

            nn.Conv1d(out_channels0, out_channels0, kernel_size=kernel_size, stride=stride, bias=True,
                      padding=padding0, dilation=dilation0),
            Chomp1d(padding0),
            nn.BatchNorm1d(out_channels0),
            nn.LeakyReLU(),
        )

        self.conv_block2 = nn.Sequential(#out_channels0
            nn.Conv1d(out_channels0, out_channels1, kernel_size=kernel_size, stride=stride, bias=True,
                       dilation=dilation1,padding=padding1), 
            Chomp1d(padding1),
            nn.BatchNorm1d(out_channels1),
            nn.LeakyReLU(),#,leaky
#
            nn.Conv1d(out_channels1, out_channels1, kernel_size=kernel_size, stride=stride, bias=True,
                      dilation=dilation1,padding=padding1), 
            Chomp1d(padding1),
            nn.BatchNorm1d(out_channels1),
            nn.LeakyReLU(),
        )
        # skip connections might be helpful to train deep networks
    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        #N: batch sizze, C_in: input dimension (1), L_in : number of conv channels
        inputs=inputs.permute(0,2,1)
        x0 = self.conv_block1(inputs) #conv_block1

        out_0 = self.relu(x0)# + res0

        x1 = self.conv_block2(out_0) #conv_block2#out_0
       
        out_1 = self.relu(x1)# + res1
        out_1=out_1.permute(0,2,1)
        
        return out_1
    
    
class BiGRU(nn.Module):
    def __init__(self, input_size, num_gru_units, hidden_size, num_layers=1):#1,100,64
        super(BiGRU, self).__init__()
        # Define the bidirectional GRU
        self.gru = nn.GRU(input_size, num_gru_units, num_layers, 
                          batch_first=True, 
                          bidirectional=True)
        # Define a fully connected layer to project the GRU output to the desired output size
        self.fc = nn.Linear(num_gru_units * 2, hidden_size)  # *2 for bidirectional, 1 if bidirec=False

    def forward(self, x):
        # Pass through GRU
        out, _ = self.gru(x)
        # Pass the output through a fully connected layer
        out = self.fc(out)
        return out
    
    
    
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__() #y:8,173,64
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        scores = self.v(torch.tanh(self.W(x)))  # Compute attention scores : 8,173,1
        weights = F.softmax(scores, dim=1).permute(0,2,1)  # Normalize scores to probabilities
        # weights: 8,173,1 = batch_size,n,m
        # y : 8,173, 64 = batch_size,m,p
        context = torch.squeeze(torch.bmm(weights, x), dim=1)  # Apply attention weights
        return context, weights
    
class ForecastingLayer(nn.Module):
    # a fully connected layer
    def __init__(self, context_dim,forecast_length):
        super(ForecastingLayer, self).__init__()
        self.fc = nn.Linear(context_dim, forecast_length)
        
    def forward(self, context):
        
        # Pass through the fully connected layer
        output = self.fc(context)  # Shape: (batch_size, forecast_length * forecast_dim)

        return output
    
class feature_extractor(nn.Module):
    def __init__(self, kernel_size,stride,in_channels0,out_channels0,in_channels1,out_channels1,input_size, num_gru_units, hidden_size, num_layers,hidden_dim):
        super(feature_extractor, self).__init__()
        self.tcn = TCN(kernel_size,stride,in_channels0,out_channels0,in_channels1,out_channels1)
        self.bigru = BiGRU(input_size, num_gru_units, hidden_size, num_layers)
        self.attn = Attention(hidden_dim)
        #self.fc = ForecastingLayer(context_dim,forecast_length)

    def forward(self, x):
        tcn_out = self.tcn(x) 
        bigru_out = self.bigru(x)  # Apply BiGRU
        out=tcn_out+bigru_out
        features, attn_weights = self.attn(out)  # Apply Attention
        #output = self.fc(features)  # Apply Fully Connected Layer
        return features,attn_weights 
    
# Full Model (Combination of all components)
class full_model(nn.Module):
    def __init__(self, kernel_size,stride,in_channels0,out_channels0,in_channels1,out_channels1,input_size, num_gru_units, hidden_size, num_layers,hidden_dim,context_dim,forecast_length):
        super(full_model, self).__init__()
        self.feature_extractor=feature_extractor(kernel_size,stride,in_channels0,out_channels0,in_channels1,out_channels1,input_size, num_gru_units, hidden_size, num_layers,hidden_dim)
        self.fc = ForecastingLayer(context_dim,forecast_length)

    def forward(self, x):
        features, attn_weights = self.feature_extractor(x)  # Apply Attention
        output = self.fc(features)  # Apply Fully Connected Layer
        return output# Return output and attention weights for explanation
    

    
device = "cuda" if torch.cuda.is_available() else 'cpu'

def train_one_epoch(full_model,
                    source_loaders,tgt_train_loader,regression_loss_function,regression_optimizer,
                    scheduler,epoch,num_epochs):
    torch.cuda.empty_cache()

    full_model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0
    num_batches=len(tgt_train_loader)
    ####### change ,src2_train_loader,src3_train_loader,src4_train_loader
    src1_train_loader,src2_train_loader,src3_train_loader,src4_train_loader=source_loaders[0],source_loaders[1],source_loaders[2],source_loaders[3]#,source_loaders[3]

##@for the source dataloaders, it depends on how many sources you have. The following line corresponds to four sources
    for (src1_batch_index, src1_batch),(src2_batch_index, src2_batch),(src3_batch_index, src3_batch),(src4_batch_index, src4_batch),(tgt_batch_index, tgt_batch)in zip(#,(src2_batch_index, src2_batch),(src3_batch_index, src3_batch),(src4_batch_index, src4_batch)
            enumerate(src1_train_loader),enumerate(src2_train_loader),enumerate(src3_train_loader),enumerate(src4_train_loader),enumerate(tgt_train_loader)):#,enumerate(src2_train_loader),enumerate(src3_train_loader),enumerate(src2_train_loader),enumerate(src3_train_loader),enumerate(src4_train_loader)
        ######################### continue //////////// ,enumerate(src4_train_loader)
        #you comment/uncomment these lines based on how many sources you have
        src1_x_batch, src1_y_batch = src1_batch[0].to(device), src1_batch[1].to(device)
        src2_x_batch, src2_y_batch = src2_batch[0].to(device), src2_batch[1].to(device)
        src3_x_batch, src3_y_batch = src3_batch[0].to(device), src3_batch[1].to(device)
        src4_x_batch, src4_y_batch = src4_batch[0].to(device), src4_batch[1].to(device)
        tgt_x_batch, tgt_y_batch = tgt_batch[0].to(device), tgt_batch[1].to(device)

        source_losses = []
        
        #@you comment/uncomment these lines based on how many sources you have
        #src1
        final_features1=full_model.feature_extractor(src1_x_batch)[0]
        prediction=full_model(src1_x_batch)
        regression_loss = regression_loss_function(prediction, src1_y_batch)

        source_losses.append(regression_loss)

        #src2
        final_features2=full_model.feature_extractor(src2_x_batch)[0]
        prediction=full_model(src2_x_batch)
        regression_loss = regression_loss_function(prediction, src2_y_batch)
        #append source features and losses 
        #source_features.append(final_features)
        source_losses.append(regression_loss)
        
#        #src3
        final_features3=full_model.feature_extractor(src3_x_batch)[0]
        prediction=full_model(src3_x_batch)
        regression_loss = regression_loss_function(prediction, src3_y_batch)
        #append source features and losses 
        #source_features.append(final_features)
        source_losses.append(regression_loss)
        ##################### uncomment this if you want a 4-source model
        #src4 #@ 
        final_features4=full_model.feature_extractor(src1_x_batch)[0]#full_model(src4_x_batch)[1] #@
        prediction=full_model(src4_x_batch)#[0] #@
        regression_loss = regression_loss_function(prediction, src4_y_batch)
        #append source features and losses 
        #source_features.append(final_features)
        source_losses.append(regression_loss)


        #@ you sum the features based on how many sources you have
        total_concatenated_features = final_features1+final_features2+final_features3+final_features4
        
        # target       
        target_features=full_model.feature_extractor(tgt_x_batch)[0]
        prediction=full_model(tgt_x_batch)
        target_loss = regression_loss_function(prediction, tgt_y_batch)

        # Compute MMD Loss, [:target_features.shape[0]]
        mmd = mmd_loss(total_concatenated_features, target_features)

        # Total loss
        total_loss = sum(source_losses) + target_loss + mmd

        # Backpropagation
        regression_optimizer.zero_grad()
        total_loss.backward()
        running_loss += target_loss.item()#mmd
        regression_optimizer.step()
        
        if (src1_batch_index==len(tgt_train_loader)-1):  
            avg_loss_across_batches = running_loss / len(tgt_train_loader) 
            print('Batch {0}, Loss: {1:.3f}'.format(src1_batch_index+1,
                                                    avg_loss_across_batches))

    
    scheduler.step()
    print()
    #@ you return the final features based on how many sources you have
    return avg_loss_across_batches,final_features1,final_features2,final_features3,final_features4,target_features#,final_features4,target_features#,final_features4#,final_features4,#,final_features3,final_features4

def validate_one_epoch(full_model,
                       epoch,valid_loader,regression_loss_function):

    full_model.train(False)
    running_loss = 0.0
    
    for batch_index, batch in enumerate(valid_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        
        with torch.no_grad():

            prediction=full_model(x_batch)
            
            loss = regression_loss_function(prediction, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(valid_loader)
    
    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('***************************************************')
    print()
    return avg_loss_across_batches

    
def evaluation(y_pred,y_true):
    rmse=np.sqrt(mean_squared_error(y_pred,y_true))
    mape=mean_absolute_percentage_error(y_pred,y_true)
    r2score=r2_score(y_pred,y_true)
    return rmse,mape,r2score



#the rbf kernel computes the similarity between each pair of samples from A and B.
# so if A(n,p) and B(n,p) then rbf_kerne(A,B) is of dimension (n,n)
#where n is the #samples & p is the #features
def rbf_kernel(A, B, bandwidth=1.0):
    # Compute the squared Euclidean distance
    A_sq = torch.sum(A**2, dim=1, keepdim=True)  # Shape (m, 1)
    B_sq = torch.sum(B**2, dim=1, keepdim=True).T  # Shape (1, n)
    
    # Using broadcasting to compute the squared distance matrix
    squared_distances = A_sq + B_sq - 2 * torch.matmul(A, B.T)
    
    # Compute the RBF kernel matrix
    K = torch.exp(-squared_distances / (2 * bandwidth**2))
    
    #rbf_kernel(target_features,target_Features) is of shape (16,16)
    #the diagonal is zeros, the other terms compute the similarity beween the different
    #target samples' features
    #then we comute the mean of similarities(see k_xx in mmd_loss)
    #the same process is done for source samples' features & between T and S
    
    return K

def mmd_loss(x, y, bandwidth=1.0): #x=source learnt features, y=target learnt features
    """Compute the MMD loss between two sets of samples using the RBF kernel."""
    # Compute the RBF kernel between x and y
    K_xx = rbf_kernel(x, x, bandwidth)  # Kernel matrix for x with itself
    K_yy = rbf_kernel(y, y, bandwidth)  # Kernel matrix for y with itself
    K_xy = rbf_kernel(x, y, bandwidth)  # Kernel matrix between x and y

    # Compute MMD
    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    #we have to minimize mmd, so minimize each term inside it, so maximize K_xy,
    # so maxinize similarity between T&S
    
    return mmd


        
def train_one_epoch_target_only(full_model,list_source_loaders,tgt_train_loader,
                    regression_loss_function,regression_optimizer,scheduler,epoch,
                    num_epochs):
    torch.cuda.empty_cache()
    full_model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0
    num_batches=len(tgt_train_loader)

    for (tgt_batch_index, tgt_batch)in enumerate(tgt_train_loader):#,enumerate(src2_train_loader),enumerate(src3_train_loader),enumerate(src2_train_loader),enumerate(src3_train_loader),enumerate(src4_train_loader)
        tgt_x_batch, tgt_y_batch = tgt_batch[0].to(device), tgt_batch[1].to(device)

        
        target_features=full_model.feature_extractor(tgt_x_batch)[0]
        prediction=full_model(tgt_x_batch)
        target_loss = regression_loss_function(prediction, tgt_y_batch)

        # Total loss
        total_loss = target_loss

        # Backpropagation
        regression_optimizer.zero_grad()
        total_loss.backward()
        running_loss += target_loss.item()
        regression_optimizer.step()
        
        if (tgt_batch_index==len(tgt_train_loader)-1):  
            avg_loss_across_batches = running_loss / len(tgt_train_loader) 
            print('Batch {0}, Loss: {1:.3f}'.format(tgt_batch_index+1,
                                                    avg_loss_across_batches))

    
    scheduler.step()
    print()
    return avg_loss_across_batches,target_features#,final_features3,final_features4

