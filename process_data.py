import pandas as pd
import glob
import os.path

fips_country_code = 'AE'
local_path = 'gdelt_data/'
# Get the GDELT field names from a helper file
colnames = pd.read_excel('CSV.header.fieldids.xlsx', sheetname='Sheet1',
                         index_col='Column ID', parse_cols=1)['Field Name']

# Build DataFrames from each of the intermediary files
files = glob.glob(local_path + 'country/' + fips_country_code + '*')
DFlist = []
for active_file in files:
    print(active_file)
    DFlist.append(pd.read_csv(active_file, sep='\t', header=None, dtype=str,
                              names=colnames, index_col=['GLOBALEVENTID']))

# Merge the file-based dataframes and save a pickle
DF = pd.concat(DFlist)
DF.to_pickle(local_path + 'backup' + fips_country_code + '.pickle')

# once everythin is safely stored away, remove the temporary files
# for active_file in files:
#     os.remove(active_file)