import csv, os
import numpy as np

def ohe_board(arr):
        OUT_SHAPE = (4,4)
        CAND = 11
        map_table = {2**i:i for i in range(1,CAND)}
        map_table[0]=0
        ret = np.zeros(shape=OUT_SHAPE + (CAND,),dtype=int)
        for r in range(OUT_SHAPE[0]):
            for c in range(OUT_SHAPE[1]):
                ret[r, c, map_table[arr[r,c]]] = 1
        return ret

# read data get x_train y_train
def read_data_select(data_file='Train.csv', n=1, begin = 4):
    reader = csv.reader(open(data_file, 'r'))
    data_list = []
    x_list = []
    y_list = []
    for one_line in reader:
        data_list.append(one_line)
    for one_line in data_list[n:]:
        one_line = list(map(int, one_line))
        length = len(one_line)
        if max(one_line) == begin: 
            b = 1
        else: 
            b = 0

        if b:
            one_list = one_line[0:length-1]
            one_list = np.array(one_list).reshape((4,4))
            one_list = ohe_board(one_list)
            x_list.append(one_list)

            tmp = np.zeros(4)    
            tmp[int(one_line[length-1])] = 1
            y_list.append(tmp)
        
    x_list = np.array(x_list)
    y_list = np.array(y_list)
    return x_list, y_list

def read_data_all(data_file='Train.csv', n=1):
    reader = csv.reader(open(data_file, 'r'))
    data_list = []
    x_list = []
    y_list = []
    for one_line in reader:
        data_list.append(one_line)
    for one_line in data_list[n:]:
        
        length = len(one_line)
        one_line = list(map(int, one_line))
        tmp = np.zeros(4)    
        tmp[one_line[length-1]] = 1
        y_list.append(tmp)
    
        one_list = one_line[0:length-1]
        one_list = np.array(one_list).reshape((4,4))
        one_list = ohe_board(one_list)
        x_list.append(one_list)
        
    x_list = np.array(x_list)
    y_list = np.array(y_list)
    return x_list, y_list