import pandas as pd
from utils import args_parser
import numpy as np
import csv
import pandas as pd
import os



# select validation dataset for test time

# ‘E:\code\fedHAR\fedhar\datasets\PAMAP\subject101

dataset_name = 'harth'
args = args_parser(dataset_name)



def client_files():
    cols=[i for i in range(args.input_dim+1)]
    w_path = 'E:\code\\fedHAR\\fedhar\datasets\\harthsp\\test4test\\'
    path = 'E:\\code\\dataset\\HAR\\har70plus\\'
    files = os.listdir(path)

    for i in range(len(files)):
        filename = path+files[i]
        df = pd.read_csv(filename, sep=',')
        data = df.values
        # set_size = int(len(df)*0.3/args.window_size)
        # test dataset
        min_length = int(len(df)*0.7)
        
        max_length = int(len(df))

        w_fn = w_path+files[i]

        batch=[]

        for j in range(int((max_length-min_length)/args.window_size)):
            i = np.random.randint(low=min_length, high=max_length-args.window_size)
            batch.extend(data[-(i+args.window_size):-i])
        
        w_df = pd.DataFrame(batch)
        w_df.to_csv(w_fn,index=False)

def global_files():
    r_path = 'E:\code\\fedHAR\\fedhar\datasets\\'+'harthsp'+'\\test4test\\'
    w_path = r_path
    files = os.listdir(r_path)
    patch = []
    for f in files:
        filename = r_path+f
        if args.header:
            df = pd.read_csv(filename)
        else:
            df = pd.read_csv(filename,header=None)
        
        data = df.values
        for s in range(20):
            start = np.random.randint(low=0,high=len(data)-args.window_size)
            part = data[start:start+args.window_size]
            patch.extend(part)
    if args.header:
        global_df = pd.DataFrame(patch,columns = df.columns)
    else:
        global_df = pd.DataFrame(patch)
    global_f = w_path + 'global.csv'
    global_df.to_csv(global_f,index=False,header=None)

global_files()

    

        
    
        
        
    

        
