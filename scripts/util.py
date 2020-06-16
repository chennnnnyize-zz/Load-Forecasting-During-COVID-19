import numpy as np


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