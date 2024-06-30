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


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class PathNetAnalyse():
    def __init__(self):
        pass

    def prepare_network(self, dataset, dataname):
        ### GET [node_num_dict] FOR WHOLE NET NODES
        kegg_gene_num_dict_df = pd.read_csv('./' + dataset + '/filtered_data/kegg_gene_num_dict.csv')
        drug_num_dict_df = pd.read_csv('./' + dataset + '/filtered_data/drug_num_dict.csv')
        kegg_gene_num_dict_df = kegg_gene_num_dict_df.rename(columns={'kegg_gene': 'node_name', 'gene_num': 'node_num'})
        kegg_gene_num_dict_df['node_type'] = ['gene'] * kegg_gene_num_dict_df.shape[0]
        drug_num_dict_df = drug_num_dict_df.rename(columns={'Drug': 'node_name', 'drug_num': 'node_num'})
        drug_num_dict_df['node_type'] = ['drug'] * drug_num_dict_df.shape[0]
        node_num_dict_df = pd.concat([kegg_gene_num_dict_df, drug_num_dict_df])
        node_num_dict_df = node_num_dict_df[['node_num', 'node_name', 'node_type']]
        node_num_dict_df.to_csv('./analysis-' + dataname + '/node_num_dict.csv', index=False, header=True)

    def sp_kth_hop_att_network(self, fold_n=1, sp=1, khop_sum=3, cell_line_name='A498', dataset='datainfo-nci', dataname='nci'):
        ##### GET CERTAIN [signaling pathway] UNDER [cell line]
        # SELECT [sp] AND [cell line]
        cell_line_map_df = pd.read_csv('./' + dataset + '/filtered_data/cell_line_map_dict.csv')
        cell_line_map_dict = dict(zip(cell_line_map_df.Cell_Line_Name, cell_line_map_df.Cell_Line_Num))
        cell_line_num = cell_line_map_dict[cell_line_name]
        sp_attention_file = './analysis-' + dataname + '/fold_' + str(fold_n) + '/sp' + str(sp) + '/cell' + str(cell_line_num)
        sp_cell_line_df = pd.read_csv(sp_attention_file + '.csv')
        # FILTER OUT ROWS WITH [mask==0]
        sp_cell_line_df.drop(sp_cell_line_df[sp_cell_line_df['Mask'] == 0.0].index, inplace=True)
        sp_cell_line_df = sp_cell_line_df.reset_index(drop=True)
        # ADD [cell line] & [signaling_path]
        kegg_sp_map_df = pd.read_csv('./' + dataset + '/filtered_data/kegg_sp_map.csv')
        kegg_sp_map_dict = dict(zip(kegg_sp_map_df.SpNotation, kegg_sp_map_df.SignalingPath))

        SpNotation = 'sp' + str(sp)
        SignalingPath = kegg_sp_map_dict[SpNotation]
        SpNotation_list = [SpNotation] * (sp_cell_line_df.shape[0])
        SignalingPath_list = [SignalingPath] * (sp_cell_line_df.shape[0])
        # REPLACE [sub_idx] WITH [node_idx]
        sp_map_file_path = './' + dataset + '/form_data/sp' + str(sp) + '_gene_map.csv'
        sp_map_df = pd.read_csv(sp_map_file_path)
        sp_map_dict = dict(zip(sp_map_df.Sub_idx, sp_map_df.Node_idx))
        sp_cell_line_df = sp_cell_line_df.replace({'From': sp_map_dict, 'To': sp_map_dict})
        sp_cell_line_df = sp_cell_line_df[['From', 'To', 'Attention', 'Hop']]
        sp_cell_line_df['SignalingPath'] = SignalingPath_list
        sp_cell_line_df['SpNotation'] = SpNotation_list
        sp_cell_line_df['Cell_Line_Name'] = [cell_line_name] * (sp_cell_line_df.shape[0])
        # REPLACE ORIGINAL FILE WITH FILTERED ONE
        sp_cell_line_df.to_csv(sp_attention_file + '_filtered.csv', index=False, header=True)
        # SEPARATE SEVERAL [cell line]
        kth_hop_sp_df_list = []
        for khop_num in range(1, khop_sum + 1):
            khop_str = 'hop' + str(khop_num)
            kth_hop_sp_df = sp_cell_line_df[sp_cell_line_df['Hop']==khop_str]
            kth_hop_sp_df.to_csv(sp_attention_file + '_' + khop_str + '.csv', index=False, header=True)
            kth_hop_sp_df_list.append(kth_hop_sp_df)
        return sp_cell_line_df, kth_hop_sp_df_list

    def organize_cell_line_specific_network(self, fold_n, dataset, dataname):
        # CELL LINE LIST
        cell_line_map_df = pd.read_csv('./' + dataset + '/filtered_data/cell_line_map_dict.csv')
        cell_line_list = list(cell_line_map_df['Cell_Line_Name'])
        cell_line_map_dict = dict(zip(cell_line_map_df.Cell_Line_Name, cell_line_map_df.Cell_Line_Num))
        # SIGNALING PATHWAY LIST
        kegg_sp_map_df = pd.read_csv('./' + dataset + '/filtered_data/kegg_sp_map.csv')
        kegg_sp_num = kegg_sp_map_df.shape[0]
        # COLLECT ALL CELL LINE SPECIFIC ATTENTION DATA FRAME
        for cell_line in cell_line_list:
            cell_specific_combined_sp_khop_df_list = []
            cell_line_num = cell_line_map_dict[cell_line]
            for sp in range(1, kegg_sp_num + 1):
                sp_cell_line_df, kth_hop_sp_df_list = PathNetAnalyse().sp_kth_hop_att_network(fold_n=fold_n, sp=sp, khop_sum=3, cell_line_name=cell_line)
                cell_specific_combined_sp_khop_df_list.append(sp_cell_line_df)
            comtemp_cell_specific_combined_sp_khop_df = pd.concat(cell_specific_combined_sp_khop_df_list)
            comtemp_cell_specific_combined_sp_khop_df.to_csv('./analysis-' + dataname + '/fold_' + str(fold_n) + '/cell' + str(cell_line_num) + '.csv', index=False, header=True)
            comtemp_cell_specific_combined_sp_khop_df.to_csv('./analysis-' + dataname + '/fold_' + str(fold_n) + '_cell/cell' + str(cell_line_num) + '.csv', index=False, header=True)
            

    def convertion_undirect_adj(self, khop_sp_cell_line_df, hop_str, sp_str, SignalingPath, cell_line_name, dataname):
        # [node_num_dict.csv]
        node_num_dict_df = pd.read_csv('./analysis-' + dataname + '/node_num_dict.csv')
        num_node = node_num_dict_df.shape[0]
        adj = np.zeros((num_node, num_node))
        # import pdb; pdb.set_trace()
        for row in khop_sp_cell_line_df.itertuples():
            row_idx = row[1] - 1
            col_idx = row[2] - 1
            attention = row[3]
            adj[row_idx, col_idx] = attention
        up_adj = np.triu(adj, k=0)
        down_adj = np.tril(adj, k=0)
        combined_adj = (up_adj + down_adj.T) / 2.0
        combined_adj = np.triu(combined_adj, k=0)
        combined_adj_sparse = sparse.csr_matrix(combined_adj)
        combined_adj_sparse = sparse_mx_to_torch_sparse_tensor(combined_adj_sparse)
        combined_adj_edgeindex = combined_adj_sparse._indices() + 1
        combined_adj_weight = combined_adj_sparse._values()
        new_khop_sp_cell_line_df = pd.DataFrame({'From': list(combined_adj_edgeindex.numpy()[0]),
                                                 'To': list(combined_adj_edgeindex.numpy()[1]),
                                                 'Attention': list(combined_adj_weight.numpy())})
        new_khop_sp_cell_line_df['Hop'] = [hop_str] * (new_khop_sp_cell_line_df.shape[0])
        new_khop_sp_cell_line_df['SignalingPath'] = [SignalingPath] * (new_khop_sp_cell_line_df.shape[0])
        new_khop_sp_cell_line_df['SpNotation'] = [sp_str] * (new_khop_sp_cell_line_df.shape[0])
        new_khop_sp_cell_line_df['Cell_Line_Name'] = [cell_line_name] * (new_khop_sp_cell_line_df.shape[0])
        new_khop_sp_cell_line_df = new_khop_sp_cell_line_df.sort_values(by = ['From', 'To'], ascending = [True, True])
        if hop_str == 'hop1':
            print(cell_line_name, hop_str, sp_str, new_khop_sp_cell_line_df.shape)
        return new_khop_sp_cell_line_df


    def convert_to_undirected(self, fold_n, dataset, dataname):
        # CELL LINE LIST
        cell_line_map_df = pd.read_csv('./' + dataset + '/filtered_data/cell_line_map_dict.csv')
        cell_line_num = cell_line_map_df.shape[0]
        cell_line_map_dict = dict(zip(cell_line_map_df.Cell_Line_Num, cell_line_map_df.Cell_Line_Name))
        # SIGNALING PATHWAY LIST
        kegg_sp_map_df = pd.read_csv('./' + dataset + '/filtered_data/cell_line_map_dict.csv')
        kegg_sp_num = kegg_sp_map_df.shape[0]
        # ADD [cell line] & [signaling_path]
        kegg_sp_map_df = pd.read_csv('./' + dataset + '/filtered_data/kegg_sp_map.csv')
        kegg_sp_map_dict = dict(zip(kegg_sp_map_df.SpNotation, kegg_sp_map_df.SignalingPath))
        # K-HOP NUMBER
        khop_sum = 3
        for cell_line in range(1, cell_line_num+1):
            cell_specific_undirected_df_list = []
            cell_line_df = pd.read_csv('./analysis-' + dataname + '/fold_' + str(fold_n) + '/cell' + str(cell_line) + '.csv')
            # import pdb; pdb.set_trace()
            for sp in range(1, kegg_sp_num+1):
                sp_str = 'sp' + str(sp)
                sp_cell_line_df = cell_line_df[cell_line_df['SpNotation']==sp_str]
                for khop in range(1, khop_sum+1):
                    hop_str = 'hop' + str(khop)
                    khop_sp_cell_line_df = sp_cell_line_df[sp_cell_line_df['Hop']==hop_str]
                    SignalingPath = kegg_sp_map_dict[sp_str]
                    cell_line_name = cell_line_map_dict[cell_line]
                    new_khop_sp_cell_line_df = PathNetAnalyse().convertion_undirect_adj(khop_sp_cell_line_df, hop_str, sp_str, SignalingPath, cell_line_name, dataname)
                    cell_specific_undirected_df_list.append(new_khop_sp_cell_line_df)
            cell_specific_undirected_df = pd.concat(cell_specific_undirected_df_list)
            cell_specific_undirected_df['From'] = cell_specific_undirected_df['From'].astype(int)
            cell_specific_undirected_df['To'] = cell_specific_undirected_df['To'].astype(int)
            cell_specific_undirected_df.to_csv('./analysis-' + dataname + '/fold_' + str(fold_n) +'_cell/cell' + str(cell_line) + '_undirected.csv', index=False, header=True)


class AverageFoldPath():
    def __init__(self):
        pass

    def average_fold_edge(self, dataname):
        ### [fold_0 is averaged path weight]
        fold_n = 0
        if os.path.exists('./analysis-' + dataname + '/fold_' + str(fold_n) +'_cell') == False:
            os.mkdir('./analysis-' + dataname + '/fold_' + str(fold_n) +'_cell')
        # CELL LINE LIST
        cell_line_map_df = pd.read_csv('./' + dataset + '/filtered_data/cell_line_map_dict.csv')
        cell_line_num = cell_line_map_df.shape[0]
        for cell_num in range(1, cell_line_num + 1):
            fold_cell_edge_df_list = []
            for fold_num in range(1, 6):
                fold_cell_path = './analysis-' + dataname + '/fold_' + str(fold_num) + '_cell/cell' + str(cell_num) +'.csv'
                fold_cell_edge_df = pd.read_csv(fold_cell_path)
                fold_cell_edge_df_list.append(fold_cell_edge_df)
            fold_cell_group_df = pd.concat(fold_cell_edge_df_list).groupby(level=0).mean()
            fold_cell_average_df = fold_cell_edge_df
            fold_cell_average_df['Attention'] = fold_cell_group_df['Attention']
            fold_cell_average_df.to_csv('./analysis-' + dataname + '/fold_' + str(fold_n) + '_cell/cell' + str(cell_num) +'.csv', index=False, header=True)
        

### DATASET SELECTION
dataset = 'datainfo-nci'
dataname = 'nci'
# dataset = 'datainfo-oneil'
# dataname = 'oneil'


# for fold_n in range(1, 6):
#     PathNetAnalyse().prepare_network(dataset, dataname)
#     if os.path.exists('./analysis-' + dataname + '/fold_' + str(fold_n) +'_cell') == False:
#         os.mkdir('./analysis-' + dataname + '/fold_' + str(fold_n) +'_cell')

#     PathNetAnalyse().organize_cell_line_specific_network(fold_n, dataset, dataname)
#     PathNetAnalyse().convert_to_undirected(fold_n, dataset, dataname)

# AverageFoldPath().average_fold_edge(dataname)