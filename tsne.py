#
# Copyright (c) 2019. Asutosh Nayak (nayak.asutosh@ymail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

"""
Final program to predict stock price. This was run on Google Colab notebook so you would find
some undefined functions like 'display' and parameters like 'PATH_TO_DRIVE_ML_DATA' which was path to my drive folder which
housed project related data. Please initialize that variable with suitable value as per your
environment.
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

stocks = pd.read_csv('stock_prices', sep=',')
zipped = list(zip(stocks["Month"], stocks["Day"], stocks["Year"]))
normalized_dates = []
for r in zipped:
    (M, D, Y) = r
    mydate = M + "," + str(D) + "," + str(Y)
    d = datetime.strptime(mydate, '%B,%d,%Y')
    normalized_dates.append(d.strftime("%Y%m%d"))

stocks['SQLDATE'] = normalized_dates
stocks['SQLDATE'] = stocks['SQLDATE'].astype(int)
# print(stocks)

fips_country_code = 'AE'
path = 'gdelt_data/backup' + fips_country_code + '.pickle'
dt = pd.read_pickle(path)
dt['SQLDATE']=dt['SQLDATE'].astype(int)
dt['EventCode'] = dt['EventCode'].astype(int)
##TODO aggregate data based on SQLDATE
# print(dt)

grouped = dt.groupby('SQLDATE').first().reset_index()
# dt2 = pd.DataFrame([], columns = ['SQLDATE', 'articles'])
dt = grouped
# for name, g in grouped:
#     dt2 = dt2.append({'SQLDATE': name, 'articles': g}, ignore_index=True)



interesting = ["SQLDATE","Actor1Code","Actor2Code","EventCode"]
final = ["Actor1Code","Actor2Code","EventCode", "Price"]
dt = dt[interesting]
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
# print(merged)
merged = merged.iloc[::-1]

print("ready")
# print(merged)


# merged['newcol'] = pd.Categorical(merged['Actor1CountryCode'])
# print(merged[displayed])
print("Input data len")
input_data = merged[final].copy()

# import talos as ta

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger("tensorflow").setLevel(logging.ERROR)
# os.environ['TZ'] = 'Asia/Kolkata'  # to set timezone; needed when running on cloud
time.tzset()

params = {
    "batch_size": 1,  # 20<16<10, 25 was a bust
    "epochs": 100,
    "lr": 0.00015000,
    "time_steps": 10
}
from sklearn.manifold import TSNE
# print(input_data)
prob = TSNE(n_components=2, perplexity=30,n_iter=3000,n_iter_without_progress=1000).fit_transform(input_data)
dataset = pd.DataFrame({'x': prob[:, 0], 'y': prob[:, 1]})
print(dataset)
merged2 = pd.concat([input_data, dataset], axis=1)
print(merged2)
from matplotlib import pyplot as plt
x,y = prob.T

plt.scatter(x,y, c=merged2["Price"])
plt.show()