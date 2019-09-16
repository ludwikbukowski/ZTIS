#
# Copyright (c) 2019. Ludwik Bukowski

"""
Final program to predict stock price.
"""

import numpy as np
import os
import sys
import time
import pandas as pd
# from tqdm._tqdm_notebook import tqdm_notebook
import pickle
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import optimizers
# from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging
import pandas as pd
import glob
import os.path
from datetime import datetime

stocks = pd.read_csv('stock_prices.csv.csv', sep=',')
zipped = list(zip(stocks["Month"], stocks["Day"], stocks["Year"]))
normalized_dates = []
for r in zipped:
    (M, D, Y) = r
    mydate = M + "," + str(D) + "," + str(Y)
    d = datetime.strptime(mydate, '%B,%d,%Y')
    normalized_dates.append(d.strftime("%Y%m%d"))

stocks['SQLDATE'] = normalized_dates
stocks['SQLDATE'] = stocks['SQLDATE'].astype(int)

fips_country_code = 'AE'
path = 'gdelt_data/backup' + fips_country_code + '.pickle'
dt = pd.read_pickle(path)
dt['SQLDATE']=dt['SQLDATE'].astype(int)
dt['EventCode'] = dt['EventCode'].astype(int)
##TODO aggregate data based on SQLDATE

grouped = dt.groupby('SQLDATE').first().reset_index()
dt = grouped



interesting = ["SQLDATE","Actor1Code","Actor2Code","EventCode"]
final = ["Actor1Code","Actor2Code","EventCode", "Price"]
dt = dt[interesting]
print("df len is " + str(len(dt)))
to_normalize = ["Actor1Code","Actor2Code"]
for f in to_normalize:
    dt[f] = dt[f].astype('category')
    dt[f] = dt[f].cat.codes
dt = dt[dt["Actor1Code"] != -1]
dt = dt[dt["Actor2Code"] != -1]

pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', 300)

merged = stocks.merge(dt, on='SQLDATE')
merged = merged[final]
merged = merged.iloc[::-1]

input_data = merged[final].copy()
# print(input_data)

logging.getLogger("tensorflow").setLevel(logging.ERROR)
time.tzset()

params = {
    "batch_size": 1,  # 20<16<10, 25 was a bust
    "epochs": 20,
    "lr": 0.00015000,
    "time_steps": 30
}

iter_changes = "dropout_layers_0.4_0.4"

OUTPUT_PATH = "outputs/lstm_best_7-3-19_12AM/"+iter_changes
TIME_STEPS = params["time_steps"]
BATCH_SIZE = params["batch_size"]
stime = time.time()


def print_time(text, stime):
    seconds = (time.time()-stime)
    print(text, seconds//60,"minutes : ",np.round(seconds%60),"seconds")


def trim_dataset(mat,batch_size):
    """
    trims dataset to a size that's divisible by BATCH_SIZE
    """
    no_of_rows_drop = mat.shape[0]%batch_size
    if no_of_rows_drop > 0:
        return mat[:-no_of_rows_drop]
    else:
        return mat


def build_timeseries(mat, y_col_index):
    # y_col_index is the index of column that would act as output column
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))

    for i in range(dim_0):
        x[i] = mat[i:TIME_STEPS+i]
        y[i] = mat[TIME_STEPS+i, y_col_index]
        # print(y[i])
    return x, y


stime = time.time()
train_cols = final
df_train, df_test = train_test_split(input_data, train_size=0.7, test_size=0.3, shuffle=False)
print("test len is " + str(len(df_test)))
print("train len is " + str(len(df_train)))
print(df_test)

min_max_scaler = MinMaxScaler()

x_train = min_max_scaler.fit_transform(df_train.loc[:,train_cols].values)
x_test = min_max_scaler.transform(df_test.loc[:,train_cols])

x_t, y_t = build_timeseries(x_train, 3)

x_test_t, y_test_t = build_timeseries(x_test, 3)

x_t = trim_dataset(x_t, BATCH_SIZE)
y_t = trim_dataset(y_t, BATCH_SIZE)

def create_model():
    lstm_model = Sequential()
    # (batch_size, timesteps, data_dim)
    lstm_model.add(LSTM(100, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]),
                        dropout=0.0, recurrent_dropout=0.0, stateful=True, return_sequences=True,
                        kernel_initializer='random_uniform'))
    lstm_model.add(Dropout(0.3))
    lstm_model.add(LSTM(100, dropout=0.001))
    lstm_model.add(Dropout(0.3))
    lstm_model.add(Dense(60,activation='relu'))
    lstm_model.add(Dense(1,activation='sigmoid'))
    optimizer = optimizers.RMSprop(lr=params["lr"])
    # optimizer = optimizers.SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)
    lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)
    return lstm_model


model = None
try:
    model = pickle.load(open("lstm_model", 'rb'))
    print("Loaded saved model...")
except FileNotFoundError:
    print("Model not found")


# x_temp, y_temp = build_timeseries(x_test, 3)

# x_temp = x_temp[:-1]
# y_temp = y_temp[:-1]
# x_val, x_test_t = np.split(trim_dataset(x_temp, BATCH_SIZE),2)
# y_val, y_test_t = np.split(trim_dataset(y_temp, BATCH_SIZE),2)

is_update_model = True
if model is None or is_update_model:
    from keras import backend as K
    print("Building model...")
    model = create_model()
    from keras.utils import plot_model
    plot_model(model, show_shapes = True, to_file='model.png')

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                       patience=40, min_delta=0.0001)
    r_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30,
                                  verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    mcp = ModelCheckpoint("best_model.h5", monitor='val_loss', verbose=1,
                          save_best_only=True, save_weights_only=False, mode='min', period=1)
    csv_logger = CSVLogger(os.path.join(OUTPUT_PATH, 'training_log_' + time.ctime().replace(" ","_") + '.log'), append=True)

    history = model.fit(x_t, y_t, epochs=params["epochs"], verbose=2, batch_size=BATCH_SIZE,
                        shuffle=False, validation_data=(trim_dataset(x_test_t, BATCH_SIZE),
                                                        trim_dataset(y_test_t, BATCH_SIZE)), callbacks=[mcp])

    print("saving model...")

    import matplotlib.pyplot as plt

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    pickle.dump(model, open("lstm_model", "wb"))

model = load_model("best_model.h5")

# res = model.evaluate(x_test_t, y_test_t, batch_size=BATCH_SIZE)
print("Predicted vs real")
y_pred = model.predict(x_test_t, batch_size=BATCH_SIZE)

y_pred_org = (y_pred * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3]
y_test_t_org = (y_test_t * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3]

print(y_pred_org)
print(y_test_t_org)

# Visualize the prediction
from matplotlib import pyplot as plt
plt.figure()
plt.plot(y_pred_org)
plt.plot(y_test_t_org)
plt.title('Prediction vs Real Stock Price')
plt.ylabel('Price')
plt.xlabel('Days')
plt.legend(['Prediction', 'Real'], loc='upper left')
plt.show()
plt.savefig(os.path.join(OUTPUT_PATH, 'newest_pred.png'))
# print_time("program completed ", stime)
