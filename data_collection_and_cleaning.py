# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 20:17:31 2024

@author: umroot
"""
import pandas as pd
from collections import Counter




def IQR_cleaning(df):
    df_copy = df.copy()
    for column in df_copy.select_dtypes(include=['number']).columns:#[1:]:#except the timestamp
        Q1 = df_copy[column].quantile(0.25)
        Q3 = df_copy[column].quantile(0.75)
        median=df_copy[column].median()
        IQR = Q3 - Q1
        df_copy[column] = df_copy[column].apply(
        lambda x: median if x < Q1-1.5*IQR or x > Q3+1.5*IQR else x
    )
    return df_copy


building_data=pd.read_csv('electricity_cleaned.txt')

#according to table1 of genome data paper, University College London (UCL) is references with 
#'robin'. The UCL site contains 52 buildings and 67 (see the last two columns of the table)
robin_columns = [col for col in building_data.columns if 'robin' in col.lower()]  
robin_columns_with_timestamp=['timestamp']+robin_columns
#extract all UCL (robin) buildings' meters (found 53 meters)
robin_meters = building_data[robin_columns_with_timestamp]
#from the 53 robin meters there are 42 meters without any NaN value
robin_meters_without_nans = robin_meters.columns[robin_meters.notna().all()].tolist()
#there are 5 types of buildings in UCL:
# Counter({'education': 23,
#          'office': 17,
#          'lodging': 10,
#          'public': 2,
#          'assembly': 1})
types_buildings=[column.split('_')[1] for column in robin_columns]#set
count = Counter(types_buildings) #givesus the count of each unique type of buildings
# we should select the same number of meterss from each type. But due to unavailability of enough
# number of meters for public&assembly we select:
# 2 public, 1 assembly, 6 education, 6 office, 6 lodging
# let's plot the curves have better idea about whihc ones to select 
# education: 'Robin_education_' + Derick,Julius,Lizbeth,Jasper,Terrance,Takako,Kristopher,Billi,
# office: 'Robin_office'+Addie, Maryann,Serena,Antonina,
# lodging: 'Robin_lodging_Dorthy',Elmer,Oliva,Celia,Renea,Janie
# public:  no one is good
# assembly: not good
selected_meters=building_data[['timestamp','Robin_education_Derick','Robin_education_Julius',
                               'Robin_education_Lizbeth','Robin_education_Jasper',
                               'Robin_education_Terrance','Robin_education_Takako',
                               'Robin_education_Kristopher','Robin_education_Billi',
                               'Robin_office_Addie','Robin_office_Maryann','Robin_office_Serena',
                               'Robin_office_Antonina',
                               'Robin_lodging_Dorthy','Robin_lodging_Elmer','Robin_lodging_Oliva',
                               'Robin_lodging_Celia','Robin_lodging_Renea','Robin_lodging_Janie']]
selected_meters_clean=IQR_cleaning(selected_meters)
selected_meters_clean.to_csv('clean_genome_meters.csv',index=False)