#Construct mobility dataset based on data from Google and Apple

import csv
import numpy as np
import pandas as pd


#Change the file name to specific mobility data
Goog_loc='Google/Seattle.csv'
Apple_loc='Apple/Seattle.csv'
Location='Seattle'

df1=pd.read_csv('%s'%Goog_loc, header=None)
df2=pd.read_csv('%s'%Apple_loc, header=None).T

#The date shift of apple data and google data based on the their report updates
start_index=34
end_index=34+91

goog_data=df1.values
apple_data=df2.values


Mobi_data=np.concatenate((apple_data, goog_data[start_index:end_index]), axis=1)




with open('%s_Mobility.csv'%Location, 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(Mobi_data)

