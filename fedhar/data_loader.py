import numpy as np
import pandas as pd
import torch
from torch.nn.functional import one_hot
from scipy import signal as signal
import os


class DataLoader:
    def __init__(self, split, cols, window_size, batch_size, args,id, header=True):
        
        train_dir=''
        test_dir =''
        # dataset init
        if args.dataset =='pamap':
            train_dir = 'datasets/PAMAP/train/'
            test_dir = 'datasets/PAMAP/test/'
            
        elif args.dataset =='harth':
            train_dir = 'datasets/HAR/harth/train/'
            test_dir = 'datasets/HAR/harth/test/'
        
        train_files = os.listdir(filename)
        filename = train_files[id]
        


        if header:
            dataframe = pd.read_csv(filename, usecols=cols, sep=',')[cols]
        else:
            dataframe = pd.read_csv(filename, header=None, sep=',', usecols=cols)[cols]
        
        self.class_num = args.class_num

        # print('dataframe col')
        # print(dataframe.columns)
        # print(dataframe.shape)

        # dataframe = dataframe[(dataframe[1] != 0)]

        print(dataframe.shape)
        dataframe.dropna(how='any', inplace=True)
        print(dataframe.shape)
        i_split = int(split * len(dataframe))
        # self.train_data = dataframe.values[:i_split]
        # 全部数据为train
        self.train_data = dataframe.values[:]

        # self.test_data = dataframe_test.values[:]
        # print(self.test_data[:,-1])

        self.train_data[:, :-1] = signal.medfilt(self.train_data[:, :-1], (3, 1))
        self.test_data[:, :-1] = signal.medfilt(self.test_data[:, :-1], (3, 1))

        self.window_size = window_size
        self.batch_size = batch_size

    def _next_window(self, start, train=True):
        if train:
            window = self.train_data[start:start + self.window_size]
        else:
            window = self.test_data[start:start + self.window_size]
        x = window[:, :-1]
        # x = x.flatten()
        y = window[-1, -1]

        return x, y

    def get_train_data(self, stride=1):
        data_x = []
        data_y = []
        for i in range(0, len(self.train_data) - self.window_size, stride):
            x, y = self._next_window(i, train=True)
            data_x.append(x)
            y = one_hot(y, num_classes=self.class_num)
            data_y.append(y)

        return np.array(data_x), np.array(data_y)

    def get_test_data(self, stride=1):
        data_x = []
        data_y = []
        for i in range(0, len(self.test_data) - self.window_size, stride):
            x, y = self._next_window(i, train=False)
            data_x.append(x)
            y = one_hot(y, num_classes=self.class_num)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def train_generator(self, stride=1):
        i = 0
        while True:
            x_batch = []
            y_batch = []
            max_length = len(self.train_data) - self.window_size
            for j in range(self.batch_size):
                i = np.random.randint(low=0, high=max_length)
                x, y = self._next_window(start=i, train=True)
                x = np.array(x)
                x_batch.append(x)
                y_batch.append(y)
            yield np.array(x_batch), np.array(y_batch)

    def test_generator(self, stride=1):
        i = 0
        while i < len(self.test_data) - self.window_size:
            x_batch = []
            y_batch = []
            for j in range(self.batch_size):
                if i >= len(self.test_data) - self.window_size:
                    break
                x, y = self._next_window(start=i, train=False)
                x = np.array(x)
                y = np.expand_dims(np.array(y))
                i += stride

            yield x,y



class DataLoaderfromfiles:
    def __init__(self,args,id, train=True):
        
        # train_dir=''
        # test_dir =''
        # dataset init
        # if args.dataset =='pamap':
        #     train_dir = 'datasets/PAMAP/train/'
        #     test_dir = 'datasets/PAMAP/test/'
            
        # elif args.dataset =='harth':
        #     train_dir = 'datasets/harth/train/'
        #     test_dir = 'datasets/PAMAP/test/'
        train_dir = 'datasets/'+args.dataset+'/train/'
        test_dir = 'datasets/'+args.dataset+'/test/'
        
        
        
        if train:
            files = os.listdir(train_dir) 
            filename = train_dir+files[id]           
        else:
            files = os.listdir(test_dir)
            filename = test_dir + files[id]
        self.file_id = files[id]

        if args.dataset == 'pamap':
            cols = [i for i in range(args.input_dim+1)]
        elif args.dataset == 'harth' or args.dataset == 'harthsp':
            cols = [i for i in range(1,args.input_dim+2)]
        if args.header:
            dataframe = pd.read_csv(filename, usecols=cols, sep=',')
        else:
            dataframe = pd.read_csv(filename, header=None, usecols=cols, sep=',')
        
        self.class_num = args.class_num

        dataframe.dropna(how='any', inplace=True)
        if args.dataset == 'harth' or args.dataset == 'harthsp':
            dataframe.drop(dataframe[dataframe['7']>args.class_num].index,inplace=True)
        # print(dataframe.shape)
        # i_split = int(split * len(dataframe))
        # self.train_data = dataframe.values[:i_split]
        # 全部数据为train
        if train:
            s_i = int(0.8*len(dataframe))
            self.train_data = dataframe.values[:s_i]
            self.test_data = dataframe.values[s_i:]
            # filter the data
            self.train_data[:, :-1] = signal.medfilt(self.train_data[:, :-1], (3, 1))
            self.test_data[:, :-1] = signal.medfilt(self.test_data[:, :-1], (3, 1))
        else:
            self.test_data = dataframe.values
            self.test_data[:, :-1] = signal.medfilt(self.test_data[:, :-1], (3, 1))

        # self.test_data = dataframe_test.values[:]
        # print(self.test_data[:,-1])

        

        self.window_size = args.window_size
        self.batch_size = args.batch_size

        self.test_index = 0

    def _next_window(self, start, train=True):
        if train:
            window = self.train_data[start:start + self.window_size]
        else:
            window = self.test_data[start:start + self.window_size]
        # window = self.data[start:start+self.window_size]
        x = window[:, :-1]
        # x = x.flatten()
        y = window[-1, -1]

        return x, y

    # def get_train_data(self, stride=1):
    #     data_x = []
    #     data_y = []
    #     for i in range(0, len(self.data) - self.window_size, stride):
    #         x, y = self._next_window(i, train=True)
    #         data_x.append(x)
    #         y = one_hot(y, num_classes=self.class_num)
    #         data_y.append(y)

    #     return np.array(data_x), np.array(data_y)

    def get_test_data(self, stride=1):
        data_x = []
        data_y = []
        for i in range(0, len(self.test_data) - self.window_size, stride):
            x, y = self._next_window(i, train=False)
            x = np.array(x)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def train_generator(self,stride=1):
        i = 0
        while True:
            x_batch = []
            y_batch = []
            max_length = len(self.train_data) - self.window_size
            for j in range(self.batch_size):
                i = np.random.randint(low=0, high=max_length)
                x, y = self._next_window(start=i, train=True)
                x = np.array(x)
                x_batch.append(x)
                y_batch.append(y)
            yield np.array(x_batch), np.array(y_batch)

    def test_generator(self, stride=100):
        # i = np.random.randint(low=0,high=len(self.test_data) - self.window_size)
        # i=0
        while self.test_index < len(self.test_data) - self.window_size:
            if self.test_index >= len(self.test_data) - self.window_size:
                break
            x, y = self._next_window(start=self.test_index, train=False)
            x = np.expand_dims(np.array(x),axis=0)
            y = np.expand_dims(np.array(y),axis=0)
            self.test_index += stride

            yield x,y

    def feedback_data(self,label):
        i = 0
        stride=100
        alter_xy = []
        alter_time = 0
        back_up_time = 0
        y_set = self.test_data[:,-1]
        freq_y = 13
        backup_xy = []
        while i < len(self.test_data) - self.window_size:
            if i >= len(self.test_data) - self.window_size:
                break
            x, y = self._next_window(start=i, train=False)

            if y ==label[0]:
                feedback_x = np.expand_dims(np.array(x),axis=0)
                feedback_y = np.expand_dims(np.array(y),axis=0)
                return feedback_x, feedback_y
            
            if y == label[1] and alter_time==0:
                alter_xy.append( np.expand_dims(np.array(x),axis=0))
                alter_xy.append(np.expand_dims(np.array(y),axis=0))
                alter_time+=1
            if y==freq_y and back_up_time==0:
                backup_xy.append( np.expand_dims(np.array(x),axis=0))
                backup_xy.append(np.expand_dims(np.array(y),axis=0))
                back_up_time+=1
            i+=stride
        if len(alter_xy) ==0:
            backup_xy.append( np.expand_dims(np.array(x),axis=0))
            backup_xy.append(np.expand_dims(np.array(y),axis=0))
            return backup_xy[0],backup_xy[1]
        return alter_xy[0],alter_xy[1]
        
            
def test4test_generator(args,f_name,global_vali=False):
    stride = 200
    test4test_path ='datasets/'+args.dataset+'/test4test/'
    
    filename = test4test_path+f_name
     
    if args.dataset == 'pamap':
        cols = [i for i in range(args.input_dim+1)]
    elif args.dataset == 'harth' or args.dataset == 'harthsp':
        cols = [i for i in range(1,args.input_dim+2)]
    
    if args.header:
        test4test_df = pd.read_csv(filename,usecols=cols)
    else:
        test4test_df = pd.read_csv(filename,header=None,usecols=cols)
    
    test_data = test4test_df.values

    i = 0
    while i < len(test_data) - args.window_size:
        window = test_data[i:i + args.window_size]
        # window = self.data[start:start+self.window_size]
        x = window[:, :-1]
        # x = x.flatten()
        y = window[-1, -1]
        x = np.expand_dims(np.array(x),axis=0)
        y = np.expand_dims(np.array(y),axis=0)
        i += stride

        yield x,y