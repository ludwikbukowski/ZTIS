import pandas as pd
import glob
import os.path
from datetime import datetime

stocks = pd.read_csv('stock_prices.csv', sep=',')
zipped = list(zip(stocks["Month"], stocks["Day"], stocks["Year"]))
normalized_dates = []
for r in zipped:
    (M, D, Y) = r
    mydate = str(M) + "," + str(D) + "," + str(Y)
    d = datetime.strptime(mydate, '%B,%d,%Y')
    normalized_dates.append(d.strftime("%Y%m%d"))

stocks['SQLDATE'] = normalized_dates
stocks['SQLDATE']=stocks['SQLDATE'].astype(int)
# print(stocks)

fips_country_code = 'AE'
path = 'gdelt_data/backup' + fips_country_code + '.pickle'
dt = pd.read_pickle(path)
dt['SQLDATE']=dt['SQLDATE'].astype(int)
# print(dt)

merged = stocks.merge(dt, on="SQLDATE")
print(merged)
