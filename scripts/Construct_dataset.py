import numpy as np
import csv
import pandas as pd

#Load data for Seattle: Jan 1st, 2018 to April 26th, 847 days in total
#Mobility data: Feb 15th to April 26th, 72 days in total
#Data is constructed by load data; weather data; Mobility data; holiday data


'''df1=pd.read_csv('../ERCOT/NCENT.csv')
val=df1.values
load_val=np.concatenate((val[:,0],val[:,1],val[:121*24,2]))
print(load_val.shape)
with open('../ERCOT/NCENT_load.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(load_val.reshape(-1, 1))'''


s='France'
num_of_days=91


Weather_data=pd.read_csv('../Europe/%s_Training3.csv'%s, header=None)
inds = pd.isnull(Weather_data).any(1).nonzero()[0]
print("Nan Val", inds)

with open('../Mobility_Data/Mobility_new/%s_Mobility.csv'%s, 'r') as csvfile:
    reader = csv.reader(csvfile)
    Mobility_data = [row for row in reader]

with open('Data_Processed/Holiday_Europe.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    Holiday_data = [row for row in reader]
Weather_data = Weather_data.values
print("Load and Weather", np.shape(Weather_data))

mobility_data = np.array(Mobility_data, dtype=float)
print("Mobility data", np.shape(mobility_data))

Holiday_data = np.array(Holiday_data, dtype=float)
print("Holiday dataset", np.shape(Holiday_data))



#index_mat:7 days, 24 hours, 1 holiday
index_mat = np.zeros((Weather_data.shape[0], 32), dtype=float)
mobility_mat = np.zeros((24*num_of_days,  mobility_data.shape[1]), dtype=float)

num=0
day=0
while num < Weather_data.shape[0]:
    for week_days in range(7):
        holiday_index = 0.0
        if Holiday_data[day] == 1:
            holiday_index = 1.0
        for hours in range(24):
            index_mat[num, (week_days+1)%7]=1.0
            index_mat[num, 7+hours]=1.0
            index_mat[num, -1]=holiday_index
            num+=1
        #if num==Weather_data.shape[0]:
        #    break
        day += 1
        if num>=Weather_data.shape[0]:
            break

training_mat = np.concatenate((Weather_data, index_mat), axis=1)
print("Whole data without mobility", np.shape(training_mat))




with open('Data_Processed_New/%s_data_all.csv'%s, 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(training_mat)

num=0
while num<num_of_days:
    for hours in range(24):
        mobility_mat[num*24+hours] = Mobility_data[num]
    num+=1


training_mat2=np.concatenate((Weather_data[-24*num_of_days:], index_mat[-24*num_of_days:], mobility_mat), axis=1)
print("Whole data with mobility", np.shape(training_mat2))

with open('Data_Processed_New/%s_mobility_all.csv'%s, 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(training_mat2)

