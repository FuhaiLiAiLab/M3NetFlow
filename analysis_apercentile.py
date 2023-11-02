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


class Percentile():
    def __init__(self):
        pass

    def gene_compute(self, n_fold, whole_net, cell_num, percentile):
        if whole_net:
            all_cell_att_deg_df = pd.read_csv('./analysis-nci/fold_' + str(n_fold) + '_cell/all_cell_whole_att_deg.csv')
        else:
            all_cell_att_deg_df = pd.read_csv('./analysis-nci/fold_' + str(n_fold) + '_cell/all_cell_att_deg.csv')

        ###### IDF APPEARANCE
        all_cell_att_matrix = all_cell_att_deg_df.iloc[:,1:].values
        all_cell_att_array = all_cell_att_matrix.flatten()
        all_cell_node_percentile = np.percentile(all_cell_att_array, percentile)
        # print('ALL CELL LINE DEGREE ' + str(percentile) + ' PERCENTILE: ', all_cell_node_percentile)

        # import pdb; pdb.set_trace()
        # CELL LINE LIST
        cell_line_map_df = pd.read_csv('./datainfo-nci/filtered_data/cell_line_map_dict.csv')
        cell_line_num = cell_line_map_df.shape[0]
        cell_line_map_dict = dict(zip(cell_line_map_df.Cell_Line_Num, cell_line_map_df.Cell_Line_Name))
        cell_name = cell_line_map_dict[cell_num]
        cell_att_deg_array = all_cell_att_deg_df[cell_name].values
        cell_node_percentile = np.percentile(cell_att_deg_array, percentile)
        # print('CELL LINE ' + cell_name + ' DEGREE ' + str(percentile) + ' DEGREE PERCENTILE: ', cell_node_percentile)
        return all_cell_node_percentile, cell_node_percentile


    def edge_compute(self, fold_n, cell_num, percentile):
        ### CELL LINE LIST
        cell_line_map_df = pd.read_csv('./datainfo-nci/filtered_data/cell_line_map_dict.csv')
        cell_line_num = cell_line_map_df.shape[0]
        cell_line_map_dict = dict(zip(cell_line_map_df.Cell_Line_Num, cell_line_map_df.Cell_Line_Name))

        # ## COMBINE ALL CELL EDGE WEIGHT
        # all_cell_path = './analysis-nci/fold_' + str(fold_n) + '_cell/cell1.csv'
        # all_cell_edge_df = pd.read_csv(all_cell_path)
        # all_cell_edge_df = all_cell_edge_df[['From', 'To', 'Hop', 'SignalingPath', 'SpNotation']]
        # for cell_num in range(1, cell_line_num + 1):
        #     cell_name = cell_line_map_dict[cell_num]
        #     fold_cell_path = './analysis-nci/fold_' + str(fold_n) + '_cell/cell' + str(cell_num) +'.csv'
        #     fold_cell_edge_df = pd.read_csv(fold_cell_path)
        #     all_cell_edge_df[cell_name] = fold_cell_edge_df['Attention']
        # all_cell_edge_df.to_csv('./analysis-nci/fold_' + str(fold_n) + '_cell/all_cell_edge.csv', index=False, header=True)

        all_cell_edge_df = pd.read_csv('./analysis-nci/fold_' + str(fold_n) + '_cell/all_cell_edge.csv')
        
        # import pdb; pdb.set_trace()
        # REWEIGHTED ALL CELL EDGES
        cell_line_list = list(cell_line_map_df['Cell_Line_Name'])
        all_cell_edge_value_df = all_cell_edge_df[cell_line_list]
        all_cell_edge_matrix = all_cell_edge_value_df.values
        all_cell_edge_attention = np.copy(all_cell_edge_matrix)
        all_cell_edge_array = all_cell_edge_matrix.flatten()
        print('ALL CELL EDGES ' + str(percentile) + ' PERCENTILE: ' , np.percentile(all_cell_edge_array, percentile))

        # import pdb; pdb.set_trace()
        cell_name = cell_line_map_dict[cell_num]
        cell_att_edge_array = all_cell_edge_df[cell_name].values
        print('CELL LINE ' + cell_name + ' EDGES ' + str(percentile) + ' PERCENTILE: ', np.percentile(cell_att_edge_array, percentile))
        return np.percentile(cell_att_edge_array, percentile)

if __name__ == "__main__":
    fold_n = 0
    percentile = 99.9
    cell_num = 26
    Percentile().gene_compute(n_fold=fold_n, whole_net=True, cell_num=cell_num, percentile=percentile)

    fold_n = 0
    percentile = 98.5
    cell_num = 26
    Percentile().edge_compute(fold_n=fold_n, cell_num=cell_num, percentile=percentile)

    # fold_n = 0
    # cell_line_map_df = pd.read_csv('./datainfo-nci/filtered_data/cell_line_map_dict.csv')
    # cell_line_num = cell_line_map_df.shape[0]
    # gene_percentile_score_list = []
    # edge_percentile_score_list = []
    # for cell_num in range(1, cell_line_num+1):
    #     gene_percentile = 99.9
    #     gene_percentile_score = Percentile().gene_compute(n_fold=fold_n, whole_net=True, cell_num=cell_num, percentile=gene_percentile)
    #     gene_percentile_score_list.append(gene_percentile_score)
    #     edge_percentile = 98.5
    #     edge_percentile_score = Percentile().edge_compute(fold_n=fold_n, cell_num=cell_num, percentile=edge_percentile)
    #     edge_percentile_score_list.append(edge_percentile_score)

    # percentile_df = pd.DataFrame({'cell_name': list(cell_line_map_df['Cell_Line_Name']),
    #                             'gene_percentile': gene_percentile_score_list, 
    #                             'edge_percentile': edge_percentile_score_list})

    # cell_line_cancer_name_map_dict_df = pd.read_csv('./datainfo-nci/filtered_data/cell_line_cancer_name_map_dict.csv')
    # percentile_df = pd.read_csv('./analysis-nci/percentile.csv')
    # percentile_df = pd.merge(percentile_df, cell_line_cancer_name_map_dict_df, how='left', left_on='cell_name', right_on='Cell_Line_Name')
    # percentile_df = percentile_df[['Cancer_name','cell_name', 'Cell_Line_Num', 'gene_percentile', 'edge_percentile']]
    # percentile_df.sort_values(by=['Cancer_name', 'cell_name']).to_csv('./analysis-nci/percentile.csv', index=False, header=True)