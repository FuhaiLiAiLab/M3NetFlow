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


class CellGeneAnalyse():
    def __init__(self):
        pass

    def combine_cell_heated_gene(self, n_fold, whole_net, dataset, dataname):
        cell_map_dict_df = pd.read_csv('./' + dataset + '/filtered_data/cell_line_map_dict.csv')
        cell_name_list = list(cell_map_dict_df['Cell_Line_Name'])
        cell_line_sum_num = cell_map_dict_df.shape[0]
        all_cell_gene_list = []
        for cell in range(1, cell_line_sum_num + 1):
            if whole_net:
                cell_path = './analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/cell' + str(cell) + '_wgc.csv'
            else:
                cell_path = './analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/cell' + str(cell) + '_gc.csv'
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
                cell_path = './analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/cell' + str(cell) + '_wgc.csv'
            else:
                cell_path = './analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/cell' + str(cell) + '_gc.csv'
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
            all_cell_gene_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/all_cell_whole_att_deg.csv', index=False, header=True)
        else:
            all_cell_gene_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/all_cell_att_deg.csv', index=False, header=True)


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
                cell_path = './analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/cell' + str(cell) + '_wgc.csv'
            else:
                cell_path = './analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/cell' + str(cell) + '_gc.csv'
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
            input_drug_effect_test_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/fold_all_whole_input_drug_target_test.csv', index=False, header=True)
        else:
            input_drug_effect_test_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/fold_all_input_drug_target_test.csv', index=False, header=True)
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
                cell_input_drug_effect_test_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/cell' + str(cell) + '_whole_input.csv', index=False, header=True)
                cell_input_drugA_num_target_effect_att_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/cell' + str(cell) + '_drugA_whole_input.csv', index=False, header=True)
                cell_input_drugB_num_target_effect_att_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/cell' + str(cell) + '_drugB_whole_input.csv', index=False, header=True)
            else:
                cell_input_drug_effect_test_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/cell' + str(cell) + '_input.csv', index=False, header=True)
                cell_input_drugA_num_target_effect_att_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/cell' + str(cell) + '_drugA_input.csv', index=False, header=True)
                cell_input_drugB_num_target_effect_att_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/cell' + str(cell) + '_drugB_input.csv', index=False, header=True)
            # import pdb; pdb.set_trace()
    

    def cell_freq_recompute(self, n_fold, whole_net, percentile, dataset, dataname):
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
        idf_freq = np.log( (all_cell_att_matrix.shape[1]+1.0) / (gene_cell_high_freq+0.01)).reshape(all_cell_att_matrix.shape[0], 1) 
        idf_freq_tile = np.tile(idf_freq, all_cell_att_matrix.shape[1])
        # RECALCULATED VALUE [idf_all_cell_att_deg]
        idf_all_cell_att_deg = np.multiply(all_cell_att_deg, idf_freq_tile)
        idf_all_cell_att_deg_df = pd.DataFrame(idf_all_cell_att_deg, columns=all_cell_att_deg_df.columns[1:])
        idf_all_cell_att_deg_df.insert(0, 'All_gene_num', all_cell_att_deg_df['All_gene_num'])
        idf_all_cell_att_deg_df.insert(0, 'node_num', np.arange(1, all_cell_att_deg_df.shape[0]+1))
        idf_all_cell_att_deg_df['freq'] = gene_cell_high_freq
        idf_all_cell_att_deg_df = idf_all_cell_att_deg_df.rename(columns={'All_gene_num': 'node_name'})
        if whole_net:
            idf_all_cell_att_deg_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/all_cell_whole_idf_att_deg.csv', index=False, header=True)
        else:
            idf_all_cell_att_deg_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/all_cell_idf_att_deg.csv', index=False, header=True)
        
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
                cell_path = './analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/cell' + str(cell) + '_wgc.csv'
            else:
                cell_path = './analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/cell' + str(cell) + '_gc.csv'
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
            input_drug_effect_test_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/fold_idf_all_whole_input_drug_target_test.csv', index=False, header=True)
        else:
            input_drug_effect_test_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/fold_idf_all_input_drug_target_test.csv', index=False, header=True)
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
                cell_input_drug_effect_test_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/cell' + str(cell) + '_whole_input_idf.csv', index=False, header=True)
                cell_input_drugA_num_target_effect_att_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/cell' + str(cell) + '_drugA_whole_input_idf.csv', index=False, header=True)
                cell_input_drugB_num_target_effect_att_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/cell' + str(cell) + '_drugB_whole_input_idf.csv', index=False, header=True)
            else:
                cell_input_drug_effect_test_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/cell' + str(cell) + '_input_idf.csv', index=False, header=True)
                cell_input_drugA_num_target_effect_att_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/cell' + str(cell) + '_drugA_input_idf.csv', index=False, header=True)
                cell_input_drugB_num_target_effect_att_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/cell' + str(cell) + '_drugB_input_idf.csv', index=False, header=True)
            # import pdb; pdb.set_trace()

    def top_cell_gene(self, n_fold, top_gene_num, dataset, dataname):
        idf_all_cell_att_deg_df = pd.read_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/all_cell_whole_idf_att_deg.csv')
        cell_line_name = idf_all_cell_att_deg_df.columns
        cell_line_name_list = list(cell_line_name)[2:]
        top_selected_gene_list = []
        top_selected_gene = []
        cell_topgene_dict = {'Cancer Name': ['Cancer'],
                    'Cell Line Name':['Cell Line'],
                    'Top1 Gene and Score':[0],
                    'Top2 Gene and Score':[0],
                    'Top3 Gene and Score':[0],
                    'Top4 Gene and Score':[0],
                    'Top5 Gene and Score':[0],
                    'Top6 Gene and Score':[0],
                    'Top7 Gene and Score':[0],
                    'Top8 Gene and Score':[0],
                    'Top9 Gene and Score':[0],
                    'Top10 Gene and Score':[0],
                    'Top11 Gene and Score':[0],
                    'Top12 Gene and Score':[0],
                    'Top13 Gene and Score':[0],
                    'Top14 Gene and Score':[0],
                    'Top15 Gene and Score':[0],
                    'Top16 Gene and Score':[0],
                    'Top17 Gene and Score':[0],
                    'Top18 Gene and Score':[0],
                    'Top19 Gene and Score':[0],
                    'Top20 Gene and Score':[0]}
        cell_topgene_df = pd.DataFrame(cell_topgene_dict)
        cancer_cell_line_dict_df = pd.read_csv('./' + dataset + '/filtered_data/cell_line_cancer_name_map_dict.csv')
        cancer_cell_line_dict = dict(zip(cancer_cell_line_dict_df.Cell_Line_Name, cancer_cell_line_dict_df.Cancer_name))
        for cell in cell_line_name_list:
            if cell == 'freq': continue
            cell_gene_rank_df = idf_all_cell_att_deg_df[['node_name', cell]]
            cell_gene_rank_df = cell_gene_rank_df.sort_values(by=[cell], ascending=False).head(top_gene_num)
            tmp_top_gene_list =  list(cell_gene_rank_df.node_name)
            tmp_top_gene_score_list = list(cell_gene_rank_df[cell])
            cell_tmp_top_gene_list = [cancer_cell_line_dict[cell]] + [cell + '_topgene_name'] + list(cell_gene_rank_df.node_name)
            cell_tmp_top_gene_score_list = [cancer_cell_line_dict[cell]] + [cell + '_topgene_score'] + list(cell_gene_rank_df[cell])
            cell_topgene_df.loc[len(cell_topgene_df.index)] = cell_tmp_top_gene_list
            cell_topgene_df.loc[len(cell_topgene_df.index)] = cell_tmp_top_gene_score_list
            top_selected_gene += tmp_top_gene_list
            top_selected_gene_list.append(tmp_top_gene_list)
            # print(cell_gene_rank_df)
        # COMMON GENES ACROSS ALL CELL LINES
        set(top_selected_gene_list[0]).intersection(*top_selected_gene_list)
        set_top_selected_gene = list(set(top_selected_gene))
        # print(top_selected_gene)
        # ORGANIZED TOP GENES IN EACH CELL LINE
        cell_topgene_df = cell_topgene_df.iloc[1: , :]
        cell_topgene_df = cell_topgene_df.sort_values(by=['Cancer Name', 'Cell Line Name'])
        print(cell_topgene_df)
        cell_topgene_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_cell/all_cell_topgene_df.csv', index=False, header=True)
        # IDENTIFY TOP GENE ACROSS CANCER CELL LINES
        cancer_list = sorted(list(set(cancer_cell_line_dict_df['Cancer_name'])))
        for cancer in cancer_list:
            tmp_cancer_df = cell_topgene_df[cell_topgene_df['Cancer Name'].isin([cancer])].reset_index(drop=True)
            tmp_cancer_df = tmp_cancer_df[tmp_cancer_df['Cell Line Name'].str.contains('topgene_name')].reset_index(drop=True)
            intersected_cancer_gene_set = set(tmp_cancer_df.iloc[0, 2:])
            for row in tmp_cancer_df.itertuples():
                intersected_cancer_gene_set = intersected_cancer_gene_set.intersection(row[2:])
            print(tmp_cancer_df)
            print(intersected_cancer_gene_set)


### DATASET SELECTION
dataset = 'datainfo-nci'
dataname = 'nci'
# dataset = 'datainfo-oneil'
# dataname = 'oneil'


# n_fold = 0
# # whole_net = True
# whole_net = False
# CellGeneAnalyse().combine_cell_heated_gene(whole_net=whole_net, n_fold=n_fold, dataset=dataset, dataname=dataname)
# CellGeneAnalyse().cell_heated_drug_target(k=5, n_fold=n_fold, whole_net=whole_net, dataset=dataset, dataname=dataname)

# # whole_net = True
# whole_net = False
# percentile = 95
# CellGeneAnalyse().cell_freq_recompute(n_fold=n_fold, whole_net=whole_net, percentile=90, dataset=dataset, dataname=dataname)

n_fold = 0
top_gene_num = 20
CellGeneAnalyse().top_cell_gene(n_fold=n_fold, top_gene_num=top_gene_num, dataset=dataset, dataname=dataname)