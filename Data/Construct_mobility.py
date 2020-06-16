import csv
import numpy as np
import pandas as pd

df1=pd.read_csv('Google/FR_Occitanie.csv', header=None)
df2=pd.read_csv('Apple/Toulouse.csv', header=None).T
df4=df2[34:34+91]


goog_data=df1.values
apple_data=df4.values

print(np.shape(goog_data))
print(np.shape(apple_data))

Mobi_data=np.concatenate((apple_data, goog_data[:91]), axis=1)




with open('Toulouse_Mobility.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(Mobi_data)

