import tensorflow as tf
import numpy as np
import keras
from keras.optimizers import SGD
import csv
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential



def nn_model(input_dim, output_dim):
    model=Sequential()
    model.add(Dense(512, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Dense(output_dim, init='normal'))
    model.add(Activation('linear'))

    return model

def calculate_MAPE(Ground_truth, pred):
    percentage_error = 0.0
    Ground_truth = np.reshape(Ground_truth, (-1,1))
    pred = np.reshape(pred, (-1, 1))
    for i in range(Ground_truth.shape[0]):
        percentage_error += np.abs(Ground_truth[i]-pred[i])/Ground_truth[i]
    percentage_error = percentage_error/Ground_truth.shape[0]

    return percentage_error



weather_dim = 5
time_dim = 32
feature_dim = weather_dim+time_dim
forecast_horizon = 1
seq_length = 24
batch_size = 32
epochs = 50


with open('Data_Processed_New/Boston_data_all.csv', 'r') as csvfile:
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


data_to_compare=np.copy(data_all_phil)




#model = lstm_model(seq_length=seq_length, input_dim=feature_dim, output_dim=forecast_horizon)
model=nn_model(input_dim=feature_dim, output_dim=forecast_horizon)
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_absolute_error', optimizer='adam')
#odel.load_weights('model/Boston_wo_Mobi.h5')
model.fit(x=training_X_phil, y=training_Y_phil, batch_size=batch_size, epochs=epochs, shuffle=True)
y_pred_large=model.predict(X_data_phil)

df_1 = pd.DataFrame(np.concatenate((y_pred_large, X_data_phil), axis=1))
pred_data = scaler_phil.inverse_transform(df_1)




print("Results for Boston")
training_MAPE=calculate_MAPE(Ground_truth=data_to_compare[:24*14,0], pred=pred_data[:24*14,0])
print("Training MAPE", training_MAPE)
testing_MAPE=calculate_MAPE(Ground_truth=data_to_compare[-24*14:,0], pred=pred_data[-24*14:,0])
print("Testing MAPE", testing_MAPE)


plt.grid()
plt.plot(pred_data[-200*14:,0],'b',label='w/o Mobility')
plt.plot(data_all_phil[-200*14:, 0],'r',label='Ground Truth')
plt.xlim([0, 200*14])
plt.xlabel('Hours')
plt.ylabel('Load (MW)')
plt.legend()
plt.title('Chicago')
plt.show()



