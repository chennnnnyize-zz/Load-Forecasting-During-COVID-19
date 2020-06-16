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
from util import *





mobility_dim=18
weather_dim=10
time_dim = 32
feature_dim = weather_dim+time_dim
feature_dim2 = weather_dim+time_dim+mobility_dim
forecast_horizon=1
seq_length=24
batch_size=32
epochs=50
num_of_days=91






with open('Data_Processed_New/UK_data_all.csv', 'r') as csvfile:
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

with open('Data_Processed_New/France_data_all.csv', 'r') as csvfile:
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

with open('Data_Processed_New/Germany_data_all.csv', 'r') as csvfile:
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







with open('Data_Processed_New/UK_mobility_all.csv', 'r') as csvfile:
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

with open('Data_Processed_New/Germany_mobility_all.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    data_all3 = [row for row in reader]

data_all3=np.array(data_all3, dtype=float)
dataset_mobi_bos = pd.DataFrame(np.copy(data_all3))
scaler_Bos_mob = StandardScaler()
scaled_columns_mobi_bos = scaler_Bos_mob.fit_transform(dataset_mobi_bos)
data_all_mobi_bos=np.array(scaled_columns_mobi_bos)

X_data_mobi_bos=data_all_mobi_bos[:, 1:]
Y_data_mobi_bos=data_all_mobi_bos[:, 0]
training_X_mobi_bos=np.copy(X_data_mobi_bos[:24*60])
training_Y_mobi_bos=np.copy(Y_data_mobi_bos[:24*60])



with open('Data_Processed_New/France_mobility_all.csv', 'r') as csvfile:
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




data_to_compare=np.copy(data_all_bos)
data_to_predict=np.copy(X_data_mobi_bos)



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
    model4.fit(x=training_X_mobi_phil, y=training_Y_mobi_phil, batch_size=batch_size, epochs=1, shuffle=True)
print("Multi-task training completed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#model4.load_weights('model/Bos_w_Mobi_Mtl.h5')
model2.fit(x=training_X_mobi_bos, y=training_Y_mobi_bos, batch_size=batch_size, epochs=2, shuffle=True)
model2.save_weights('model/France_w_Mobi_Mtl.h5')
#The prediction results for multi-task learning
y_pred_mobi_mtl= model3.predict(data_to_predict)




#model = lstm_model(seq_length=seq_length, input_dim=feature_dim, output_dim=forecast_horizon)
model=nn_model(input_dim=feature_dim, output_dim=forecast_horizon)
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_absolute_error', optimizer='adam')
#odel.load_weights('model/Boston_wo_Mobi.h5')
model.fit(x=training_X_bos, y=training_Y_bos, batch_size=batch_size, epochs=epochs, shuffle=True)
model.save_weights('model/SA_wo_Mobi.h5')
y_pred_large=model.predict(X_data_chi)




model_small=nn_model(input_dim=feature_dim, output_dim=forecast_horizon)
#predictions = model(x)  #Works for nn and rnn model
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model_small.compile(loss='mean_absolute_error', optimizer='adam')
#model.load_weights('model/Phil_wo_Mobi.h5')
model_small.fit(x=training_X_mobi_bos[:, :feature_dim], y=training_Y_mobi_bos, batch_size=batch_size, epochs=epochs, shuffle=True)
#model.save_weights('model/Phil_wo_Mobi.h5')
y_pred_small=model.predict(data_to_predict[:,:feature_dim])





'''plt.grid()
plt.plot(pred_data[:24*14,0],'b',label='w/o Mobility')
plt.plot(data_all[:24*14:, 0],'r',label='Ground Truth')
plt.xlim([0, 24*14])
plt.xlabel('Hours')
plt.ylabel('Load (MW)')
plt.legend()
plt.show()'''



'''zx = predictions(shared(Dense(512, activation='relu', name='x_limb')(x)))
zy = predictions(shared(Dense(512, activation='relu', name='y_limb')(y)))

model_x = Model(inputs=[x], outputs=[zx])
model_y = Model(inputs=[y], outputs=[zy])
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model_x.compile(loss='mean_squared_error', optimizer='adam')
print("Training begins")
model_x.fit(x=training_X_mobi, y=training_Y_mobi, batch_size=batch_size, epochs=epochs, shuffle=True)'''





model_sole=nn_model(input_dim=feature_dim2, output_dim=forecast_horizon)
#predictions = model(x)  #Works for nn and rnn model
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model_sole.compile(loss='mean_absolute_error', optimizer='adam')


#model3.load_weights('model/Chicago_w_Mobi.h5')
model_sole.fit(x=training_X_mobi_bos, y=training_Y_mobi_bos, batch_size=batch_size, epochs=epochs, shuffle=True)
model_sole.save_weights('model/SA_w_Mobi.h5')
y_pred_mobi_sole_model = model_sole.predict(data_to_predict)






df_1 = pd.DataFrame(np.concatenate((y_pred_large, X_data_bos), axis=1))
pred_data = scaler_bos.inverse_transform(df_1)
df_single_mob = pd.DataFrame(np.concatenate((y_pred_mobi_sole_model, data_to_predict), axis=1))
pred_data_mobi_sole = scaler_Bos_mob.inverse_transform(df_single_mob)
df_mtl = pd.DataFrame(np.concatenate((y_pred_mobi_mtl, data_to_predict), axis=1))
pred_data_mobi_mtl = scaler_Bos_mob.inverse_transform(df_mtl)
df_small = pd.DataFrame(np.concatenate((y_pred_small, data_to_predict), axis=1))
pred_data_small = scaler_Bos_mob.inverse_transform(df_small)




print("Results for Chicago")
training_MAPE=calculate_MAPE(Ground_truth=data_to_compare[:24*14,0], pred=pred_data[:24*14,0])
print("Training MAPE NN_Orig", training_MAPE)
testing_MAPE=calculate_MAPE(Ground_truth=data_to_compare[-24*14:,0], pred=pred_data[-24*14:,0])
print("Testing MAPE NN_Orig", testing_MAPE)
training_MAPE=calculate_MAPE(Ground_truth=data_to_compare[:24*14,0], pred=pred_data_small[-24*num_of_days:-24*14, 0])
print("Training MAPE Retrain", training_MAPE)
testing_MAPE=calculate_MAPE(Ground_truth=data_to_compare[-24*14:,0], pred=pred_data_small[-24*14:, 0])
print("Testing MAPE Retrain", testing_MAPE)
training_MAPE=calculate_MAPE(Ground_truth=data_to_compare[-24*num_of_days:-24*14, 0], pred=pred_data_mobi_sole[-24*num_of_days:-24*14, 0])
print("Training MAPE Mobility", training_MAPE)
testing_MAPE=calculate_MAPE(Ground_truth=data_to_compare[-24*14:, 0], pred=pred_data_mobi_sole[-24*14:, 0])
print("Testing MAPE Mobility", testing_MAPE)
pred_data_mobi_mtl=(pred_data_mobi_mtl[-24*num_of_days:, 0]).reshape(-1, 1)
training_MAPE=calculate_MAPE(Ground_truth=data_to_compare[-24*num_of_days:-24*14, 0], pred=pred_data_mobi_mtl[-24*num_of_days:-24*14, 0])
print("Training MAPE MTL", training_MAPE)
testing_MAPE=calculate_MAPE(Ground_truth=data_to_compare[-24*14:, 0], pred=pred_data_mobi_mtl[-24*14:, 0])
print("Testing MAPE MTL", testing_MAPE)


results=np.concatenate((data_all_bos[-24*num_of_days:, 0].reshape(-1, 1),
                        pred_data[-24*num_of_days:, 0].reshape(-1, 1),
                        pred_data_small[-24*num_of_days:, 0].reshape(-1, 1),
                        pred_data_mobi_sole[-24*num_of_days:, 0].reshape(-1, 1),
                        pred_data_mobi_mtl[-24 * num_of_days:, 0].reshape(-1, 1)), axis=1)
with open('Results/Germany_Test_Results.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(results)

from matplotlib.pyplot import figure
figure(num=None, figsize=(9, 6), dpi=80, facecolor='w', edgecolor='k')


plt.grid()
plt.plot(pred_data[-24*14:,0],'b',label='NN_Orig')
plt.plot(pred_data_mobi_sole[-24*14:,0],'g', label='With Mobility Single Model')
plt.plot(pred_data_mobi_mtl[-24*14:,0], 'y', label='With Mobility MTL Model')
plt.plot(data_all_bos[-24*14:, 0],'r',label='Ground Truth')
plt.plot(pred_data_small[-24*14:, 0],'c',label='Retrain')
plt.xlim([0, 24*14])
plt.xlabel('Hours')
plt.ylabel('Load (MW)')
plt.legend()
plt.show()



