import numpy as np
from sklearn.preprocessing import MinMaxScaler


SEQUENCE_LENGTH = 100


def scale_data(train_data, test_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_data)
    scaled_test = scaler.transform(test_data)
    return scaler, scaled_train, scaled_test


def create_sequences(data):
    x = []
    y = []
    
    for i in range(SEQUENCE_LENGTH, len(data)):
        x.append(data[i-SEQUENCE_LENGTH:i])
        y.append(data[i])

    x = np.array(x)
    y = np.array(y)

    return x, y