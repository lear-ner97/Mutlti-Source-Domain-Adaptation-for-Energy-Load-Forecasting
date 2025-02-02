# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 22:43:57 2024

@author: umroot
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


Julius_data = pd.read_csv('clean_genome_meters.csv', sep=',')[['timestamp','Robin_education_Julius']]
Lizbeth_data = pd.read_csv('clean_genome_meters.csv', sep=',')[['timestamp','Robin_education_Lizbeth']]
Jasper_data = pd.read_csv('clean_genome_meters.csv', sep=',')[['timestamp','Robin_education_Jasper']]
Terrance_data = pd.read_csv('clean_genome_meters.csv', sep=',')[['timestamp','Robin_education_Terrance']]
Takako_data = pd.read_csv('clean_genome_meters.csv', sep=',')[['timestamp','Robin_education_Takako']]
Kristopher_data = pd.read_csv('clean_genome_meters.csv', sep=',')[['timestamp','Robin_education_Kristopher']]
Billi_data = pd.read_csv('clean_genome_meters.csv', sep=',')[['timestamp','Robin_education_Billi']]



period = Julius_data['timestamp']
period = pd.to_datetime(period)


tgt_period = Billi_data['timestamp']
tgt_period = pd.to_datetime(tgt_period)


plt.figure(figsize=(8,6))
plt.plot(period[8785:10200], Julius_data['Robin_education_Julius'][8785:10200])#[-len(new_y_test):]
plt.plot(period[8785:10200], Lizbeth_data['Robin_education_Lizbeth'][8785:10200])
plt.plot(tgt_period[8785:10200], Billi_data['Robin_education_Billi'][8785:10200])
#plt.plot(test_time[-test_time_range:][:tgt_horizon*7], test_predictions[::tgt_horizon, :].flatten()[:tgt_horizon*7], label='model',color='orange')  # modif_tf[:tgt_horizon*7]
# plt.plot(test_time[-test_time_range:][:tgt_horizon*7],test_predictions_cnn_lstm[::tgt_horizon,:].flatten()[:tgt_horizon*7], label='CNN-LSTM DAF',color='green')
# plt.plot(test_time[-test_time_range:][:tgt_horizon*7],test_predictions_tf[::tgt_horizon,:].flatten()[:tgt_horizon*7], label='Traditional TF DAF',color='blue')
plt.xlabel('time')
plt.ylabel('load (kWh)')
plt.gcf().autofmt_xdate() 
# plt.title('testing')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
#plt.savefig(tgt_building+str(weeks)+'.png', bbox_inches='tight')
plt.show()