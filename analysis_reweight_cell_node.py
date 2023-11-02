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


class ReweightPathCellGeneAnalyse():
    def __init__(self):
        pass

    def combine_cell_heated_gene(self, n_fold, whole_net, dataset, dataname):
        cell_map_dict_df = pd.read_csv('./' + dataset + '/filtered_data/cell_line_map_dict.csv')
        cell_name_list = list(cell_map_dict_df['Cell_Line_Name'])
        cell_line_sum_num = cell_map_dict_df.shape[0]
        all_cell_gene_list = []
        for cell in range(1, cell_line_sum_num + 1):
            if whole_net:
                cell_path = './analysis-' + dataname + '/fold_' + str(n_fold) + '_cell-w/reweight_cell' + str(cell) + '_wgc.csv'
            else:
                cell_path = './analysis-' + dataname + '/fold_' + str(n_fold) + '_cell-w/reweight_cell' + str(cell) + '_gc.csv'
            cell_num_gc_df = pd.read_csv(cell_path)
            cell_node = list(cell_num_gc_df['node_num'])
            all_cell_gene_list += cell_node
        # COMBINE ALL CELL [att_deg] TO ONE DATAFRAME
        all_cell_gene_list = sorted(list(set(all_cell_gene_list)))
        all_cell_gene_df = pd.DataFrame({'All_gene_num': all_cell_gene_list})
        all_cell_gene_col_list = ['All_gene_num']
        for cell in range(1, cell_line_sum_num + 1):
            # FECTH EACH CELL LINE'S GENE [att_deg]
            if whole_net:
                cell_path = './analysis-' + dataname + '/fold_' + str(n_fold) + '_cell-w/reweight_cell' + str(cell) + '_wgc.csv'
            else:
                cell_path = './analysis-' + dataname + '/fold_' + str(n_fold) + '_cell-w/reweight_cell' + str(cell) + '_gc.csv'
            cell_num_gc_df = pd.read_csv(cell_path)
            cell_num_gc_df = cell_num_gc_df[['node_num', 'att_deg']]
            # MERGE FOR [all_cell_gene_df]
            # cell_att_deg = 'cell'+str(cell)+'_att_deg'
            cell_name = cell_name_list[cell-1]
            cell_att_deg = cell_name
            all_cell_gene_col_list.append(cell_att_deg)
            cell_num_gc_df = cell_num_gc_df.rename(columns={'att_deg': cell_att_deg})
            all_cell_gene_df = pd.merge(all_cell_gene_df, cell_num_gc_df, how='left', left_on='All_gene_num', right_on='node_num')
            all_cell_gene_df = all_cell_gene_df[all_cell_gene_col_list]
        all_cell_gene_df = all_cell_gene_df.fillna(0.0)
        # REPLACE [node_num] WITH GENE NAME
        node_num_dict_df = pd.read_csv('./analysis-' + dataname + '/node_num_dict.csv')
        node_map_dict = dict(zip(node_num_dict_df.node_num, node_num_dict_df.node_name))
        all_cell_gene_df = all_cell_gene_df.replace({'All_gene_num': node_map_dict})
        if whole_net:
            all_cell_gene_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell-w/reweight_all_cell_whole_att_deg.csv', index=False, header=True)
        else:
            all_cell_gene_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell-w/reweight_all_cell_att_deg.csv', index=False, header=True)


    def cell_heated_drug_target(self, k, n_fold, whole_net, dataset, dataname):
        n_fold_train_input_df_list = []
        for i in range(1, k+1):
            if i==n_fold: continue
            else:
                n_split_input_df = pd.read_csv('./' + dataset + '/filtered_data/split_input_' + str(i) + '.csv')
                n_fold_train_input_df_list.append(n_split_input_df)
        # n_fold_train_input_df = pd.concat(n_fold_train_input_df_list).reset_index(drop=True)
        n_fold_train_input_df = pd.read_csv('./' + dataset + '/filtered_data/random_final_dl_input.csv')
        # import pdb; pdb.set_trace()
        ##### CREATE ONE_HOT FOR [CELL DRUG-TARGET] PAIRS
        cell_map_dict_df = pd.read_csv('./' + dataset + '/filtered_data/cell_line_map_dict.csv')
        cell_name_list = list(cell_map_dict_df['Cell_Line_Name'])
        cell_dict = dict(zip(cell_name_list, range(1, len(cell_name_list)+1)))
        cell_num_dict = dict(zip(range(1, len(cell_name_list)+1), cell_name_list))
        node_num_dict_df = pd.read_csv('./analysis-' + dataname + '/node_num_dict.csv')
        node_dict = dict(zip(node_num_dict_df.node_name, node_num_dict_df.node_num))
        node_num_dict = dict(zip(node_num_dict_df.node_num, node_num_dict_df.node_name))
        final_drugbank_num_df = pd.read_csv('./' + dataset + '/filtered_data/final_drugbank_num.csv')
        # BUILD CELL LINE CORE NODE LIST (DO NOT NEED TO LOAD DATA EVERYTIME)
        cell_num_gc_df_list = []
        for cell in range(1, len(cell_name_list) + 1):
            # FECTH EACH CELL LINE'S GENE [att_deg]
            if whole_net:
                cell_path = './analysis-' + dataname + '/fold_' + str(n_fold) + '_cell-w/reweight_cell' + str(cell) + '_wgc.csv'
            else:
                cell_path = './analysis-' + dataname + '/fold_' + str(n_fold) + '_cell-w/reweight_cell' + str(cell) + '_gc.csv'
            cell_num_gc_df = pd.read_csv(cell_path)
            cell_num_gc_df_list.append(cell_num_gc_df)
        # BUILD UP JUDGEMENT ON EACH DATA POINT
        input_drug_effect_test = np.zeros((n_fold_train_input_df.shape[0], 8))
        for row in n_fold_train_input_df.itertuples():
            drugA_idx = node_dict[row[1]]
            drugB_idx = node_dict[row[2]]
            cell_idx = cell_dict[row[3]]
            score = row[4]
            cell_num_gc_df = cell_num_gc_df_list[cell_idx - 1]
            cell_num_gc_gene_list = list(cell_num_gc_df['node_num'])
            # DrugA
            drugA_target_df = final_drugbank_num_df[final_drugbank_num_df['Drug']==drugA_idx]
            drugA_target_df = drugA_target_df[drugA_target_df['Target'].isin(cell_num_gc_gene_list)].reset_index(drop=True)
            drugA_target_num = drugA_target_df.shape[0]
            drugA_target_att_df = pd.merge(drugA_target_df, cell_num_gc_df, how='left', left_on='Target', right_on='node_num')
            drugA_target_att_sum = drugA_target_att_df['att_deg'].sum()
            # if cell_idx == 1 and drugA_idx==1495:
            #     import pdb; pdb.set_trace()
            # DrugB
            drugB_target_df = final_drugbank_num_df[final_drugbank_num_df['Drug']==drugB_idx]
            drugB_target_df = drugB_target_df[drugB_target_df['Target'].isin(cell_num_gc_gene_list)].reset_index(drop=True)
            drugB_target_num = drugB_target_df.shape[0]
            drugB_target_att_df = pd.merge(drugB_target_df, cell_num_gc_df, how='left', left_on='Target', right_on='node_num')
            drugB_target_att_sum = drugB_target_att_df['att_deg'].sum()
            # INSERT INTO TEST
            input_drug_effect_test[row[0], 0] = drugA_idx
            input_drug_effect_test[row[0], 1] = drugA_target_num
            input_drug_effect_test[row[0], 2] = drugA_target_att_sum
            input_drug_effect_test[row[0], 3] = drugB_idx
            input_drug_effect_test[row[0], 4] = drugB_target_num
            input_drug_effect_test[row[0], 5] = drugB_target_att_sum
            input_drug_effect_test[row[0], 6] = cell_idx
            input_drug_effect_test[row[0], 7] = score
        input_drug_effect_test_df = pd.DataFrame(input_drug_effect_test, 
                            columns=['DrugA_idx', 'DrugA_target_num', 'DrugA_target_att_sum',
                                     'DrugB_idx', 'DrugB_target_num', 'DrugB_target_att_sum',
                                     'Cell_idx', 'Score'])
        input_drug_num_effect_test_df = input_drug_effect_test_df      
        input_drug_num_effect_test_df = input_drug_num_effect_test_df.replace({'Cell_idx': cell_num_dict})        
        input_drug_effect_test_df = input_drug_effect_test_df.replace({'DrugA_idx': node_num_dict, 
                                                'DrugB_idx': node_num_dict, 'Cell_idx': cell_num_dict})
        if whole_net:
            input_drug_effect_test_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell-w/fold_reweight_all_whole_input_drug_target_test.csv', index=False, header=True)
        else:
            input_drug_effect_test_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell-w/fold_reweight_all_input_drug_target_test.csv', index=False, header=True)
        # DIVIDED BY CELL LINE
        for cell in range(1, len(cell_name_list) + 1):
            # FECTH EACH CELL LINE
            cell_name = cell_num_dict[cell]
            cell_input_drug_effect_test_df = input_drug_effect_test_df[input_drug_effect_test_df['Cell_idx']==cell_name]
            cell_input_drug_effect_test_df = cell_input_drug_effect_test_df.sort_values(by=['Score'], ascending=False)
            # FECTH EACH CELL LINE FOR EACH DRUG
            cell_num_gc_df = cell_num_gc_df_list[cell - 1]
            cell_input_drug_num_effect_test_df = input_drug_num_effect_test_df[input_drug_num_effect_test_df['Cell_idx']==cell_name]
            cell_input_drug_num_effect_test_df['index1'] = cell_input_drug_num_effect_test_df.index
            # DrugA
            cell_input_drugA_num_target_effect_test_df = pd.merge(cell_input_drug_num_effect_test_df, final_drugbank_num_df, how='left', left_on='DrugA_idx', right_on='Drug')
            cell_input_drugA_num_target_effect_att_df = pd.merge(cell_input_drugA_num_target_effect_test_df, cell_num_gc_df, how='left', left_on='Target', right_on='node_num')
            cell_input_drugA_num_target_effect_att_df = cell_input_drugA_num_target_effect_att_df.sort_values(by=['Score'], ascending=False)
            # DrugB
            cell_input_drugB_num_target_effect_test_df = pd.merge(cell_input_drug_num_effect_test_df, final_drugbank_num_df, how='left', left_on='DrugB_idx', right_on='Drug')
            cell_input_drugB_num_target_effect_att_df = pd.merge(cell_input_drugB_num_target_effect_test_df, cell_num_gc_df, how='left', left_on='Target', right_on='node_num')
            cell_input_drugB_num_target_effect_att_df = cell_input_drugB_num_target_effect_att_df.sort_values(by=['Score'], ascending=False)
            # import pdb; pdb.set_trace()
            if whole_net:
                cell_input_drug_effect_test_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell-w/reweight_cell' + str(cell) + '_whole_input.csv', index=False, header=True)
                cell_input_drugA_num_target_effect_att_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell-w/reweight_cell' + str(cell) + '_drugA_whole_input.csv', index=False, header=True)
                cell_input_drugB_num_target_effect_att_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell-w/reweight_cell' + str(cell) + '_drugB_whole_input.csv', index=False, header=True)
            else:
                cell_input_drug_effect_test_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell-w/reweight_cell' + str(cell) + '_input.csv', index=False, header=True)
                cell_input_drugA_num_target_effect_att_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell-w/reweight_cell' + str(cell) + '_drugA_input.csv', index=False, header=True)
                cell_input_drugB_num_target_effect_att_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell-w/reweight_cell' + str(cell) + '_drugB_input.csv', index=False, header=True)
            # import pdb; pdb.set_trace()
    

    def cell_freq_recompute(self, n_fold, whole_net, percentile, dataname):
        if whole_net:
            all_cell_att_deg_df = pd.read_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/all_cell_whole_att_deg.csv')
        else:
            all_cell_att_deg_df = pd.read_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/all_cell_att_deg.csv')

        ###### IDF APPEARANCE
        all_cell_att_matrix = all_cell_att_deg_df.iloc[:,1:].values
        all_cell_att_array = all_cell_att_matrix.flatten()
        all_cell_att_matrix[all_cell_att_matrix < np.percentile(all_cell_att_array, percentile)] = 0
        all_cell_att_matrix[all_cell_att_matrix >= np.percentile(all_cell_att_array, percentile)] = 1
        gene_cell_high_freq = np.sum(all_cell_att_matrix, axis=1)

        ##### IDF-Freq
        # ORIGINAL VALUE [all_cell_att_deg]
        all_cell_att_deg = all_cell_att_deg_df.iloc[:,1:].values
        idf_freq = np.log( (all_cell_att_matrix.shape[1]+1) / (gene_cell_high_freq+0.5)).reshape(all_cell_att_matrix.shape[0], 1) 
        idf_freq_tile = np.tile(idf_freq, all_cell_att_matrix.shape[1])
        # RECALCULATED VALUE [idf_all_cell_att_deg]
        idf_all_cell_att_deg = np.multiply(all_cell_att_deg, idf_freq_tile)
        idf_all_cell_att_deg_df = pd.DataFrame(idf_all_cell_att_deg, columns=all_cell_att_deg_df.columns[1:])
        idf_all_cell_att_deg_df.insert(0, 'All_gene_num', all_cell_att_deg_df['All_gene_num'])
        idf_all_cell_att_deg_df.insert(0, 'node_num', np.arange(1, all_cell_att_deg_df.shape[0]+1))
        idf_all_cell_att_deg_df['freq'] = gene_cell_high_freq
        idf_all_cell_att_deg_df = idf_all_cell_att_deg_df.rename(columns={'All_gene_num': 'node_name'})

        ##### IDF VERSION
        n_fold_train_input_df = pd.read_csv('./' + dataset + '/filtered_data/random_final_dl_input.csv')
        ##### CREATE ONE_HOT FOR [CELL DRUG-TARGET] PAIRS
        cell_map_dict_df = pd.read_csv('./' + dataset + '/filtered_data/cell_line_map_dict.csv')
        cell_name_list = list(cell_map_dict_df['Cell_Line_Name'])
        cell_dict = dict(zip(cell_name_list, range(1, len(cell_name_list)+1)))
        cell_num_dict = dict(zip(range(1, len(cell_name_list)+1), cell_name_list))
        node_num_dict_df = pd.read_csv('./analysis-' + dataname + '/node_num_dict.csv')
        node_dict = dict(zip(node_num_dict_df.node_name, node_num_dict_df.node_num))
        node_num_dict = dict(zip(node_num_dict_df.node_num, node_num_dict_df.node_name))
        final_drugbank_num_df = pd.read_csv('./' + dataset + '/filtered_data/final_drugbank_num.csv')

        # BUILD CELL LINE CORE NODE LIST (DO NOT NEED TO LOAD DATA EVERYTIME)
        cell_num_gc_df_list = []
        for cell in range(1, len(cell_name_list) + 1):
            # FECTH EACH CELL LINE'S GENE [att_deg]
            if whole_net:
                cell_path = './analysis-' + dataname + '/fold_' + str(n_fold) + '_cell-w/reweight_cell' + str(cell) + '_wgc.csv'
            else:
                cell_path = './analysis-' + dataname + '/fold_' + str(n_fold) + '_cell-w/reweight_cell' + str(cell) + '_gc.csv'
            cell_num_gc_df = pd.read_csv(cell_path)
            cell_num_gc_df_list.append(cell_num_gc_df)
        # BUILD UP JUDGEMENT ON EACH DATA POINT
        input_drug_effect_test = np.zeros((n_fold_train_input_df.shape[0], 8))
        for row in n_fold_train_input_df.itertuples():
            drugA_idx = node_dict[row[1]]
            drugB_idx = node_dict[row[2]]
            cell_idx = cell_dict[row[3]]
            cell_name = row[3]
            score = row[4]
            cell_num_gc_df = idf_all_cell_att_deg_df[['node_num', 'node_name', cell_name]]
            cell_num_gc_df = cell_num_gc_df.rename(columns={cell_name: 'att_deg'})
            cell_num_gc_gene_list = list(cell_num_gc_df['node_num'])
            # DrugA
            drugA_target_df = final_drugbank_num_df[final_drugbank_num_df['Drug']==drugA_idx]
            drugA_target_df = drugA_target_df[drugA_target_df['Target'].isin(cell_num_gc_gene_list)].reset_index(drop=True)
            drugA_target_num = drugA_target_df.shape[0]
            drugA_target_att_df = pd.merge(drugA_target_df, cell_num_gc_df, how='left', left_on='Target', right_on='node_num')
            drugA_target_att_sum = drugA_target_att_df['att_deg'].sum()
            # if cell_idx == 1 and drugA_idx==1495:
            #     import pdb; pdb.set_trace()
            # DrugB
            drugB_target_df = final_drugbank_num_df[final_drugbank_num_df['Drug']==drugB_idx]
            drugB_target_df = drugB_target_df[drugB_target_df['Target'].isin(cell_num_gc_gene_list)].reset_index(drop=True)
            drugB_target_num = drugB_target_df.shape[0]

            drugB_target_att_df = pd.merge(drugB_target_df, cell_num_gc_df, how='left', left_on='Target', right_on='node_num')
            drugB_target_att_sum = drugB_target_att_df['att_deg'].sum()
            # INSERT INTO TEST
            input_drug_effect_test[row[0], 0] = drugA_idx
            input_drug_effect_test[row[0], 1] = drugA_target_num
            input_drug_effect_test[row[0], 2] = drugA_target_att_sum
            input_drug_effect_test[row[0], 3] = drugB_idx
            input_drug_effect_test[row[0], 4] = drugB_target_num
            input_drug_effect_test[row[0], 5] = drugB_target_att_sum
            input_drug_effect_test[row[0], 6] = cell_idx
            input_drug_effect_test[row[0], 7] = score
        input_drug_effect_test_df = pd.DataFrame(input_drug_effect_test, 
                            columns=['DrugA_idx', 'DrugA_target_num', 'DrugA_target_idfatt_sum',
                                     'DrugB_idx', 'DrugB_target_num', 'DrugB_target_idfatt_sum',
                                     'Cell_idx', 'Score'])
        input_drug_num_effect_test_df = input_drug_effect_test_df      
        input_drug_num_effect_test_df = input_drug_num_effect_test_df.replace({'Cell_idx': cell_num_dict})        
        input_drug_effect_test_df = input_drug_effect_test_df.replace({'DrugA_idx': node_num_dict, 
                                                'DrugB_idx': node_num_dict, 'Cell_idx': cell_num_dict})
        if whole_net:
            input_drug_effect_test_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell-w/fold_reweight_idf_all_whole_input_drug_target_test.csv', index=False, header=True)
        else:
            input_drug_effect_test_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell-w/fold_reweight_idf_all_input_drug_target_test.csv', index=False, header=True)
        # DIVIDED BY CELL LINE
        for cell in range(1, len(cell_name_list) + 1):
            # FECTH EACH CELL LINE
            cell_name = cell_num_dict[cell]
            cell_input_drug_effect_test_df = input_drug_effect_test_df[input_drug_effect_test_df['Cell_idx']==cell_name]
            cell_input_drug_effect_test_df = cell_input_drug_effect_test_df.sort_values(by=['Score'], ascending=False)
            # FECTH EACH CELL LINE FOR EACH DRUG
            cell_num_gc_df = idf_all_cell_att_deg_df[['node_num', 'node_name', cell_name]]
            cell_num_gc_df = cell_num_gc_df.rename(columns={cell_name: 'att_deg'})
            cell_input_drug_num_effect_test_df = input_drug_num_effect_test_df[input_drug_num_effect_test_df['Cell_idx']==cell_name]
            cell_input_drug_num_effect_test_df['index1'] = cell_input_drug_num_effect_test_df.index
            # DrugA
            cell_input_drugA_num_target_effect_test_df = pd.merge(cell_input_drug_num_effect_test_df, final_drugbank_num_df, how='left', left_on='DrugA_idx', right_on='Drug')
            cell_input_drugA_num_target_effect_att_df = pd.merge(cell_input_drugA_num_target_effect_test_df, cell_num_gc_df, how='left', left_on='Target', right_on='node_num')
            cell_input_drugA_num_target_effect_att_df = cell_input_drugA_num_target_effect_att_df.sort_values(by=['Score'], ascending=False)
            # DrugB
            cell_input_drugB_num_target_effect_test_df = pd.merge(cell_input_drug_num_effect_test_df, final_drugbank_num_df, how='left', left_on='DrugB_idx', right_on='Drug')
            cell_input_drugB_num_target_effect_att_df = pd.merge(cell_input_drugB_num_target_effect_test_df, cell_num_gc_df, how='left', left_on='Target', right_on='node_num')
            cell_input_drugB_num_target_effect_att_df = cell_input_drugB_num_target_effect_att_df.sort_values(by=['Score'], ascending=False)
            # import pdb; pdb.set_trace()
            if whole_net:
                cell_input_drug_effect_test_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell-w/reweight_cell' + str(cell) + '_whole_input_idf.csv', index=False, header=True)
                cell_input_drugA_num_target_effect_att_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell-w/reweight_cell' + str(cell) + '_drugA_whole_input_idf.csv', index=False, header=True)
                cell_input_drugB_num_target_effect_att_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell-w/reweight_cell' + str(cell) + '_drugB_whole_input_idf.csv', index=False, header=True)
            else:
                cell_input_drug_effect_test_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell-w/reweight_cell' + str(cell) + '_input_idf.csv', index=False, header=True)
                cell_input_drugA_num_target_effect_att_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell-w/reweight_cell' + str(cell) + '_drugA_input_idf.csv', index=False, header=True)
                cell_input_drugB_num_target_effect_att_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell-w/reweight_cell' + str(cell) + '_drugB_input_idf.csv', index=False, header=True)
            # import pdb; pdb.set_trace()

### DATASET SELECTION
dataset = 'datainfo-nci'
dataname = 'nci'

n_fold = 0
# whole_net = True
# ReweightPathCellGeneAnalyse().combine_cell_heated_gene(whole_net=whole_net, n_fold=n_fold)
# ReweightPathCellGeneAnalyse().cell_heated_drug_target(k=5, n_fold=n_fold, whole_net=whole_net)

whole_net = True
percentile = 90
ReweightPathCellGeneAnalyse().cell_freq_recompute(n_fold=n_fold, whole_net=whole_net, percentile=90, dataname=dataname)