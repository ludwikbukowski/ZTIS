import requests
import lxml.html as lh
import os.path
import urllib
import zipfile
import glob
import operator
import glob
import pandas as pd

gdelt_base_url = 'http://data.gdeltproject.org/events/'
infilecounter = 0
outfilecounter = 0
# get the list of all the links on the gdelt file page
page = requests.get(gdelt_base_url+'index.html')
doc = lh.fromstring(page.content)
link_list = doc.xpath("//*/ul/li/a/@href")

# separate out those links that begin with four digits 
file_list = [x for x in link_list if str.isdigit(x[0:4])]
local_path = 'gdelt_data/'

fips_country_code = 'AE'
interested_in = ['13', '30']

def filter_row(line, fpscode):
    splited = line.split('\t')
    geocode =  fpscode in operator.itemgetter(51, 37, 44)(splited)
    action =  operator.itemgetter(28)(splited) in interested_in
    # if action:
    #     print("Found oune!")
    return geocode and action



for compressed_file in file_list[infilecounter:]:
    print(compressed_file)
    
    # if we dont have the compressed file stored locally, go get it. Keep trying if necessary.
    while not os.path.isfile(local_path+compressed_file): 
        # print('downloading,')
        urllib.request.urlretrieve(url=gdelt_base_url+compressed_file,
                           filename=local_path+compressed_file)
        
    # extract the contents of the compressed file to a temporary directory    
    # print('extracting,')
    z = zipfile.ZipFile(file=local_path+compressed_file, mode='r')    
    z.extractall(path=local_path+'tmp/')
    
    # parse each of the csv files in the working directory, 
    # print('parsing,')
    for infile_name in glob.glob(local_path+'tmp/*'):
        outfile_name = local_path+'country/'+fips_country_code+'%04i.tsv'%outfilecounter
        
        # open the infile and outfile
        with open(infile_name, mode='r') as infile, open(outfile_name, mode='w') as outfile:
            for line in infile:
                # extract lines with our interest country code
                if filter_row(line, fips_country_code):
                    outfile.write(line)
            outfilecounter +=1
            
        # delete the temporary file
        os.remove(infile_name)
    infilecounter +=1
    # print('done')



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
for active_file in files:
    os.remove(active_file)