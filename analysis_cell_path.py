import os
import pdb
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import sparse
from matplotlib.pyplot import figure
from sklearn.preprocessing import normalize
from matplotlib.ticker import PercentFormatter

from pandas import DataFrame
import seaborn as sns

class AnalysisCellPath():
    def __init__(self):
        pass

    def cell_path(self, dataset):
        cell_path_df = pd.read_csv('./analysis-' + dataset + '/cancer_cell_att_sp-paper.csv', encoding='latin-1')
        cell_path_df = cell_path_df[['Cancer_name', 'Cell_Line_Name', 'SignalingPath', 'sp_num_edges', 'all_num_edges', 'edge_weight_ratio', 'validation-papers']]
        cell_path_df.loc[~cell_path_df['validation-papers'].str.contains('--'), 'validation-papers'] = 'validated'
        cell_path_df.loc[cell_path_df['validation-papers'].str.contains('--'), 'validation-papers'] = 'non-validated'

        ### CONSTRUCT [cancer, cell, signalingpath] NODE MAP DICT
        cancer_name_list = sorted(list(set(cell_path_df['Cancer_name'])))
        cell_name_list = sorted(list(set(cell_path_df['Cell_Line_Name'])))
        sp_name_list = sorted(list(set(cell_path_df['SignalingPath'])))
        all_node_list = cancer_name_list + cell_name_list + sp_name_list
        all_node_idx_list = list(np.arange(1, len(all_node_list)+1))
        cancer_type_list = ['Cancer'] * len(cancer_name_list)
        cell_type_list = ['Cell'] * len(cell_name_list)
        sp_type_list = ['SignalingPath'] * len(sp_name_list)
        type_list = cancer_type_list + cell_type_list + sp_type_list
        cancer_cell_path_df = pd.DataFrame({'Node_idx': all_node_idx_list, 'Node_name': all_node_list, 'Type_name': type_list})
        cancer_cell_path_df.to_csv('./analysis-nci/cancer_cell_path_name_dict.csv', index=False, header=True)
        all_node_dict = dict(zip(all_node_list, all_node_idx_list))

        ### CONSTRUCT [cancer-cell, cell-signalingpath] EDGE
        #
        cancer_cell_df = cell_path_df[['Cancer_name', 'Cell_Line_Name']]
        cancer_cell_df['Edge_type'] = ['cancer-cell'] * (cancer_cell_df.shape[0])
        cancer_cell_df['Validation'] = ['truth'] * (cancer_cell_df.shape[0])
        cancer_cell_df = cancer_cell_df.drop_duplicates().reset_index(drop = True)
        cancer_cell_df.rename(columns = {'Cancer_name':'From', 'Cell_Line_Name':'To'}, inplace = True)
        cancer_cell_df = cancer_cell_df.replace({'From': all_node_dict, 'To': all_node_dict})
        #
        cell_sp_df = cell_path_df[['Cell_Line_Name', 'SignalingPath', 'validation-papers']]
        cell_sp_df = cell_sp_df.drop_duplicates().reset_index(drop = True)
        cell_sp_df.rename(columns = {'Cell_Line_Name':'From', 'SignalingPath':'To', 'validation-papers': 'Validation'}, inplace = True)
        cell_sp_df = cell_sp_df.replace({'From': all_node_dict, 'To': all_node_dict})
        cell_sp_df['Edge_type'] = ['cell-sp'] * (cell_sp_df.shape[0])
        cancer_cell_sp_df = pd.concat([cancer_cell_df, cell_sp_df], ignore_index=True)
        cancer_cell_sp_df.to_csv('./analysis-nci/cancer_cell_sp_edge.csv', index=False, header=True)


dataset = 'nci'
AnalysisCellPath().cell_path(dataset)