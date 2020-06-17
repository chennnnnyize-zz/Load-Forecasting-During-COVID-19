import tensorflow as tf
import numpy as np
import keras
from keras.optimizers import SGD
import csv
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from nn_model import nn_model
from keras import Model, Input
from keras.layers import Dense





def calculate_MAPE(Ground_truth, pred):
    percentage_error = 0.0
    Ground_truth = np.reshape(Ground_truth, (-1,1))
    pred = np.reshape(pred, (-1, 1))
    for i in range(Ground_truth.shape[0]):
        percentage_error += np.abs(Ground_truth[i]-pred[i])/Ground_truth[i]
    percentage_error = percentage_error/Ground_truth.shape[0]

    return percentage_error


def reorganize2(X_train, Y_train, seq_length, forecast_horizon, forecast_time):
    # Organize the input and output to feed into RNN model
    x_data = []
    y_data=[]
    for i in range(len(X_train) - seq_length-forecast_horizon-forecast_time):
        x_new = X_train[i:i + seq_length]
        y_new = Y_train[i:i + seq_length].reshape(-1,1)
        x_new = np.concatenate((x_new, y_new), axis=1)
        x_data.append(x_new)
        y_new = Y_train[i+forecast_time-1+seq_length:i+forecast_horizon+forecast_time+seq_length-1].\
            reshape(-1, forecast_horizon)
        y_data.append(y_new)

    return x_data, y_data


mobility_dim=9
weather_dim=5
time_dim = 32
feature_dim = weather_dim+time_dim
feature_dim2 = weather_dim+time_dim+mobility_dim
forecast_horizon=1
seq_length=24
batch_size=32
epochs=50
num_of_days=91






with open('Data_Processed_New/Chicago_data_all.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    data_all = [row for row in reader]

data_all=np.array(data_all, dtype=float)
dataset_chi = pd.DataFrame(np.copy(data_all))
scaler_Chi = StandardScaler()
scaled_columns_chi = scaler_Chi.fit_transform(dataset_chi)
data_all_wo_mobi_chi=np.array(scaled_columns_chi)

X_data_chi=data_all_wo_mobi_chi[:, 1:]
Y_data_chi=data_all_wo_mobi_chi[:, 0]
training_X_chi=np.copy(X_data_chi[:24*730])
training_Y_chi=np.copy(Y_data_chi[:24*730])

with open('Data_Processed_New/Phil_data_all.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    data_all_phil = [row for row in reader]

data_all_phil=np.array(data_all_phil, dtype=float)
dataset_phil = pd.DataFrame(np.copy(data_all_phil))
scaler_phil = StandardScaler()
scaled_columns_phil = scaler_phil.fit_transform(dataset_phil)
data_all_wo_mobi_phil = np.array(scaled_columns_phil)

X_data_phil = data_all_wo_mobi_phil[:, 1:]
Y_data_phil = data_all_wo_mobi_phil[:, 0]
training_X_phil = np.copy(X_data_phil[:24*730])
training_Y_phil = np.copy(Y_data_phil[:24*730])

with open('Data_Processed_New/Boston_data_all.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    data_all_bos = [row for row in reader]

data_all_bos=np.array(data_all_bos, dtype=float)
dataset_bos = pd.DataFrame(np.copy(data_all_bos))
scaler_bos = StandardScaler()
scaled_columns_bos = scaler_bos.fit_transform(dataset_bos)
data_all_wo_mobi_bos=np.array(scaled_columns_bos)

X_data_bos=data_all_wo_mobi_bos[:, 1:]
Y_data_bos=data_all_wo_mobi_bos[:, 0]
training_X_bos=np.copy(X_data_bos[:24*730])
training_Y_bos=np.copy(Y_data_bos[:24*730])

with open('Data_Processed_New/Seattle_data_all.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    data_all_sea = [row for row in reader]

data_all_sea=np.array(data_all_sea, dtype=float)
dataset_sea = pd.DataFrame(np.copy(data_all_sea))
scaler_sea = StandardScaler()
scaled_columns_sea = scaler_sea.fit_transform(dataset_sea)
data_all_wo_mobi_sea=np.array(scaled_columns_sea)

X_data_sea=data_all_wo_mobi_sea[:, 1:]
Y_data_sea=data_all_wo_mobi_sea[:, 0]
training_X_sea=np.copy(X_data_sea[:24*730])
training_Y_sea=np.copy(Y_data_sea[:24*730])

print("Training x sea", np.shape(X_data_sea))





with open('Data_Processed_New/Chicago_mobility_all.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    data_all2 = [row for row in reader]

data_all2=np.array(data_all2, dtype=float)
dataset_mobi_chi = pd.DataFrame(np.copy(data_all2))
scaler_Chicago_mob = StandardScaler()
scaled_columns_mobi = scaler_Chicago_mob.fit_transform(dataset_mobi_chi)
data_all_mobi=np.array(scaled_columns_mobi)

X_data_mobi_chi=data_all_mobi[:, 1:]
Y_data_mobi_chi=data_all_mobi[:, 0]
training_X_mobi_chi=np.copy(X_data_mobi_chi)
training_Y_mobi_chi=np.copy(Y_data_mobi_chi)

with open('Data_Processed_New/Boston_mobility_all.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    data_all3 = [row for row in reader]

data_all3=np.array(data_all3, dtype=float)
dataset_mobi_bos = pd.DataFrame(np.copy(data_all3))
scaler_Bos_mob = StandardScaler()
scaled_columns_mobi_bos = scaler_Bos_mob.fit_transform(dataset_mobi_bos)
data_all_mobi_bos=np.array(scaled_columns_mobi_bos)

X_data_mobi_bos=data_all_mobi_bos[:, 1:]
Y_data_mobi_bos=data_all_mobi_bos[:, 0]
training_X_mobi_bos=np.copy(X_data_mobi_bos)
training_Y_mobi_bos=np.copy(Y_data_mobi_bos)

with open('Data_Processed_New/Seattle_mobility_all.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    data_all4 = [row for row in reader]

data_all4=np.array(data_all4, dtype=float)
print("DAta ", np.shape(data_all4))
data_all44=np.concatenate((dataset_sea, np.tile(data_all4[0, -9:], data_all_wo_mobi_sea.shape[0]).reshape(-1, 9)), axis=1)
data_all44=np.concatenate((data_all44, data_all4), axis=0)
dataset_mobi_sea = pd.DataFrame(np.copy(data_all44))
print(data_all44[0])
print(data_all4[0])
scaler_Sea_mob = StandardScaler()
scaled_columns_mobi_sea = scaler_Sea_mob.fit_transform(dataset_mobi_sea)
data_all_seasea=np.array(scaled_columns_mobi_sea)

data_all_mobi_sea=np.array(scaled_columns_mobi_sea[-2184:,:])

X_data_mobi_sea=data_all_mobi_sea[:, 1:]
Y_data_mobi_sea=data_all_mobi_sea[:, 0]
training_X_mobi_sea=np.copy(X_data_mobi_sea)
training_Y_mobi_sea=np.copy(Y_data_mobi_sea)
print("Training data SIZE", np.shape(training_X_mobi_sea))
mobillity_mean=np.mean(X_data_mobi_sea[24*80:, -9:], axis=0)
mobillity_var=np.std(X_data_mobi_sea[24*70:, -9:], axis=0)
mobillity_max=np.mean(X_data_mobi_sea[:24*7, -9:], axis=0)
print("mobility mean", mobillity_mean)
print("mobility var", mobillity_var)
print("mobility max", np.max(X_data_mobi_sea[24*70:, -9:], axis=0))


#######################################The code for plots#################################################
forecast1_upper=np.concatenate((data_all_seasea[:, 1:38], np.tile(mobillity_mean+1.96*mobillity_var, data_all_seasea.shape[0]).reshape(-1, 9)), axis=1)
forecast1_lower=np.concatenate((data_all_seasea[:, 1:38], np.tile(mobillity_mean-1.96*mobillity_var, data_all_seasea.shape[0]).reshape(-1, 9)), axis=1)
forecast2_upper=np.concatenate((data_all_seasea[:, 1:38], np.tile(mobillity_max+1.96*mobillity_var, data_all_seasea.shape[0]).reshape(-1, 9)), axis=1)
forecast2_lower=np.concatenate((data_all_seasea[:, 1:38], np.tile(mobillity_max-1.96*mobillity_var, data_all_seasea.shape[0]).reshape(-1, 9)), axis=1)
forecast1=np.concatenate((data_all_seasea[:, 1:38], np.tile(mobillity_mean, data_all_seasea.shape[0]).reshape(-1, 9)), axis=1)
forecast2=np.concatenate((data_all_seasea[:, 1:38], np.tile(mobillity_max, data_all_seasea.shape[0]).reshape(-1, 9)), axis=1)


with open('Data_Processed_New/Phil_mobility_all.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    data_all5 = [row for row in reader]

data_all5=np.array(data_all5, dtype=float)
dataset_mobi_phil = pd.DataFrame(np.copy(data_all5))
scaler_Phil_Mob = StandardScaler()
scaled_columns_mobi_phil = scaler_Phil_Mob.fit_transform(dataset_mobi_phil)
data_all_mobi_phil=np.array(scaled_columns_mobi_phil)

X_data_mobi_phil=data_all_mobi_phil[:, 1:]
Y_data_mobi_phil=data_all_mobi_phil[:, 0]
training_X_mobi_phil=np.copy(X_data_mobi_phil)
training_Y_mobi_phil=np.copy(Y_data_mobi_phil)




data_to_compare=np.copy(data_all_sea)
data_to_predict=np.copy(X_data_mobi_sea)



x = Input(shape=(feature_dim2,))

shared = Dense(128, activation='relu', name='shared')
predictions = Dense(forecast_horizon, activation='softmax', name='predictions')
layer1 = Dense(512, activation='relu')(x)
layer2 = Dense(128, activation='relu')(layer1)
layer3 = Dense(32, activation='relu')(layer2)
output11 = Dense(16)(layer3)
output1 = Dense(forecast_horizon)(output11)
output22 = Dense(forecast_horizon)(layer3)
output2 = Dense(forecast_horizon)(output22)
output33 = Dense(forecast_horizon)(layer3)
output3 = Dense(forecast_horizon)(output33)
output44 = Dense(forecast_horizon)(layer3)
output4 = Dense(forecast_horizon)(output44)
model1 = Model(input=x, output=output1)
model1.compile(loss='mean_absolute_error', optimizer='adam')
model2 = Model(input=x, output=output2)
model2.compile(loss='mean_absolute_error', optimizer='adam')
model3 = Model(input=x, output=output3)
model3.compile(loss='mean_absolute_error', optimizer='adam')
model4 = Model(input=x, output=output4)
model4.compile(loss='mean_absolute_error', optimizer='adam')

for epoch_num in range(50):
    model1.fit(x=training_X_mobi_chi, y=training_Y_mobi_chi, batch_size=batch_size, epochs=1, shuffle=True)
    model2.fit(x=training_X_mobi_bos, y=training_Y_mobi_bos, batch_size=batch_size, epochs=1, shuffle=True)
    model3.fit(x=training_X_mobi_sea, y=training_Y_mobi_sea, batch_size=batch_size, epochs=1, shuffle=True)
    model4.fit(x=training_X_mobi_phil, y=training_Y_mobi_phil, batch_size=batch_size, epochs=1, shuffle=True)
print("Multi-task training completed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
model3.fit(x=training_X_mobi_sea, y=training_Y_mobi_sea, batch_size=batch_size, epochs=20, shuffle=True)
model3.save_weights('model/Seattle_w_Mobi_Mtl.h5')
#The prediction results for multi-task learning
y_pred_mobi_mtl= model3.predict(data_to_predict)



y_pred_mobi_sole_model = model3.predict(data_to_predict)
df_single_mob = pd.DataFrame(np.concatenate((y_pred_mobi_sole_model, data_to_predict), axis=1))
pred_data_mobi_sole = scaler_Sea_mob.inverse_transform(df_single_mob)

y_pred_sea_2020_high = model3.predict(forecast1_upper)
df_1 = pd.DataFrame(np.concatenate((y_pred_sea_2020_high, forecast1_lower), axis=1))
pred_data = scaler_Sea_mob.inverse_transform(df_1)
y_pred_sea_2020_low=model3.predict(forecast1_lower)
df_2 = pd.DataFrame(np.concatenate((y_pred_sea_2020_low, forecast1_lower), axis=1))
pred_data2 = scaler_Sea_mob.inverse_transform(df_2)
y_pred_sea_2020_high2=model3.predict(forecast2_upper)
df_3 = pd.DataFrame(np.concatenate((y_pred_sea_2020_high2, forecast1_lower), axis=1))
pred_data3 = scaler_Sea_mob.inverse_transform(df_3)
y_pred_sea_2020_low2=model3.predict(forecast2_lower)
df_4 = pd.DataFrame(np.concatenate((y_pred_sea_2020_low2, forecast1_lower), axis=1))
pred_data4 = scaler_Sea_mob.inverse_transform(df_4)

y_pred_sea_2020 = model3.predict(forecast1)
df_11 = pd.DataFrame(np.concatenate((y_pred_sea_2020, forecast1_lower), axis=1))
pred_data11 = scaler_Sea_mob.inverse_transform(df_11)
y_pred_sea_20202 = model3.predict(forecast2)
df_22 = pd.DataFrame(np.concatenate((y_pred_sea_20202, forecast1_lower), axis=1))
pred_data22 = scaler_Sea_mob.inverse_transform(df_22)





pred_ALL=np.concatenate((pred_data[:,0].reshape(-1, 1), pred_data2[:,0].reshape(-1, 1),pred_data11[:,0].reshape(-1, 1),
                         pred_data3[:,0].reshape(-1, 1), pred_data4[:,0].reshape(-1, 1), pred_data22[:,0].reshape(-1, 1),), axis=1)


















