import os
import re
import numpy as np
import pandas as pd

from load_data import LoadData

# RANDOMIZE THE [final_NCI60_DeepLearningInput]
def input_random(dataset):
    final_input_df = pd.read_csv('./' +dataset + '/filtered_data/final_dl_input.csv')
    random_final_input_df = final_input_df.sample(frac = 1)
    random_final_input_df.to_csv('./' +dataset + '/filtered_data/random_final_dl_input.csv', index = False, header = True)

# SPLIT DEEP LEARNING INPUT INTO TRAINING AND TEST
def split_k_fold(k, dataset):
    random_final_dl_input_df = pd.read_csv('./' +dataset + '/filtered_data/random_final_dl_input.csv')
    num_points = random_final_dl_input_df.shape[0]
    num_div = int(num_points / k)
    num_div_list = [i * num_div for i in range(0, k)]
    num_div_list.append(num_points)
    # SPLIT [RandomFinal_NCI60_DeepLearningInput] INTO [k] FOLDS
    for place_num in range(k):
        low_idx = num_div_list[place_num]
        high_idx = num_div_list[place_num + 1]
        print('\n--------TRAIN-TEST SPLIT WITH TEST FROM ' + str(low_idx) + ' TO ' + str(high_idx) + '--------')
        split_input_df = random_final_dl_input_df[low_idx : high_idx]
        split_input_df.to_csv('./' +dataset + '/filtered_data/split_input_' + str(place_num + 1) + '.csv', index = False, header = True)

### DATASET SELECTION
# dataset = 'datainfo-nci'
dataset = 'datainfo-oneil'

# input_random(dataset)
# split_k_fold(k=5, dataset=dataset)

# if os.path.exists('./' +dataset + '/form_data') == False:
#     os.mkdir('./' +dataset + '/form_data')
# k = 5
# batch_size = 64
# LoadData().load_all_split(batch_size, k, dataset)

# ############## MOUDLE 2 ################
# LoadData().load_edge_adj(dataset)
# LoadData().load_adj_edgeindex(dataset)
# LoadData().load_path_adj_edgeindex(dataset=dataset)
# LoadData().load_path_adj_khop_mask_edgeindex(khop_num=3, dataset=dataset)

################ MOUDLE 3 ################
# FORM N-TH FOLD TRAINING DATASET
k = 5
n_fold = 5
LoadData().load_train_test(k, n_fold, dataset)