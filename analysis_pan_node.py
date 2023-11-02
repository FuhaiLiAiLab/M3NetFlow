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


class PanGeneAnalyse():
    def __init__(self):
        pass

    def pan_drug_target(self, k, n_fold, whole_net, dataset, dataname):
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
        # FECTH EACH PAN'S GENE [att_deg]
        if whole_net:
            pan_path = './analysis-' + dataname + '/fold_' + str(n_fold) + '_pan/pan_wgc.csv'
        else:
            pan_path = './analysis-' + dataname + '/fold_' + str(n_fold) + '_pan/pan_gc.csv'
        pan_gc_df = pd.read_csv(pan_path)
        # BUILD UP JUDGEMENT ON EACH DATA POINT
        input_drug_effect_test = np.zeros((n_fold_train_input_df.shape[0], 8))
        for row in n_fold_train_input_df.itertuples():
            drugA_idx = node_dict[row[1]]
            drugB_idx = node_dict[row[2]]
            cell_idx = cell_dict[row[3]]
            score = row[4]
            pan_gc_gene_list = list(pan_gc_df['node_num'])
            # DrugA
            drugA_target_df = final_drugbank_num_df[final_drugbank_num_df['Drug']==drugA_idx]
            drugA_target_df = drugA_target_df[drugA_target_df['Target'].isin(pan_gc_gene_list)].reset_index(drop=True)
            drugA_target_num = drugA_target_df.shape[0]
            drugA_target_att_df = pd.merge(drugA_target_df, pan_gc_df, how='left', left_on='Target', right_on='node_num')
            drugA_target_att_sum = drugA_target_att_df['att_deg'].sum()
            # if cell_idx == 1 and drugA_idx==1495:
            #     import pdb; pdb.set_trace()
            # DrugB
            drugB_target_df = final_drugbank_num_df[final_drugbank_num_df['Drug']==drugB_idx]
            drugB_target_df = drugB_target_df[drugB_target_df['Target'].isin(pan_gc_gene_list)].reset_index(drop=True)
            drugB_target_num = drugB_target_df.shape[0]
            drugB_target_att_df = pd.merge(drugB_target_df, pan_gc_df, how='left', left_on='Target', right_on='node_num')
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
            input_drug_effect_test_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_pan/fold_all_whole_input_drug_target_test.csv', index=False, header=True)
            # DrugA
            input_drugA_num_target_effect_test_df = pd.merge(input_drug_num_effect_test_df, final_drugbank_num_df, how='left', left_on='DrugA_idx', right_on='Drug')
            input_drugA_num_target_effect_att_df = pd.merge(input_drugA_num_target_effect_test_df, pan_gc_df, how='left', left_on='Target', right_on='node_num')
            input_drugA_num_target_effect_att_df = input_drugA_num_target_effect_att_df.sort_values(by=['Score'], ascending=False)
            # DrugB
            input_drugB_num_target_effect_test_df = pd.merge(input_drug_num_effect_test_df, final_drugbank_num_df, how='left', left_on='DrugB_idx', right_on='Drug')
            input_drugB_num_target_effect_att_df = pd.merge(input_drugB_num_target_effect_test_df, pan_gc_df, how='left', left_on='Target', right_on='node_num')
            input_drugB_num_target_effect_att_df = input_drugB_num_target_effect_att_df.sort_values(by=['Score'], ascending=False)
            input_drugA_num_target_effect_att_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_pan/fold_all_whole_input_drugA_whole_input.csv', index=False, header=True)
            input_drugB_num_target_effect_att_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_pan/fold_all_whole_input_drugB_whole_input.csv', index=False, header=True)
        else:
            input_drug_effect_test_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_pan/fold_all_input_drug_target_test.csv', index=False, header=True)
            # DrugA
            input_drugA_num_target_effect_test_df = pd.merge(input_drug_num_effect_test_df, final_drugbank_num_df, how='left', left_on='DrugA_idx', right_on='Drug')
            input_drugA_num_target_effect_att_df = pd.merge(input_drugA_num_target_effect_test_df, pan_gc_df, how='left', left_on='Target', right_on='node_num')
            input_drugA_num_target_effect_att_df = input_drugA_num_target_effect_att_df.sort_values(by=['Score'], ascending=False)
            # DrugB
            input_drugB_num_target_effect_test_df = pd.merge(input_drug_num_effect_test_df, final_drugbank_num_df, how='left', left_on='DrugB_idx', right_on='Drug')
            input_drugB_num_target_effect_att_df = pd.merge(input_drugB_num_target_effect_test_df, pan_gc_df, how='left', left_on='Target', right_on='node_num')
            input_drugB_num_target_effect_att_df = input_drugB_num_target_effect_att_df.sort_values(by=['Score'], ascending=False)
            input_drugA_num_target_effect_att_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_pan/fold_all_input_drugA_whole_input.csv', index=False, header=True)
            input_drugB_num_target_effect_att_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_pan/fold_all_input_drugB_whole_input.csv', index=False, header=True)

        # DIVIDED BY CELL LINE
        for cell in range(1, len(cell_name_list) + 1):
            # FECTH EACH CELL LINE
            cell_name = cell_num_dict[cell]
            cell_input_drug_effect_test_df = input_drug_effect_test_df[input_drug_effect_test_df['Cell_idx']==cell_name]
            cell_input_drug_effect_test_df = cell_input_drug_effect_test_df.sort_values(by=['Score'], ascending=False)
            # FECTH EACH CELL LINE FOR EACH DRUG
            cell_input_drug_num_effect_test_df = input_drug_num_effect_test_df[input_drug_num_effect_test_df['Cell_idx']==cell_name]
            cell_input_drug_num_effect_test_df['index1'] = cell_input_drug_num_effect_test_df.index
            # DrugA
            cell_input_drugA_num_target_effect_test_df = pd.merge(cell_input_drug_num_effect_test_df, final_drugbank_num_df, how='left', left_on='DrugA_idx', right_on='Drug')
            cell_input_drugA_num_target_effect_att_df = pd.merge(cell_input_drugA_num_target_effect_test_df, pan_gc_df, how='left', left_on='Target', right_on='node_num')
            cell_input_drugA_num_target_effect_att_df = cell_input_drugA_num_target_effect_att_df.sort_values(by=['Score'], ascending=False)
            # DrugB
            cell_input_drugB_num_target_effect_test_df = pd.merge(cell_input_drug_num_effect_test_df, final_drugbank_num_df, how='left', left_on='DrugB_idx', right_on='Drug')
            cell_input_drugB_num_target_effect_att_df = pd.merge(cell_input_drugB_num_target_effect_test_df, pan_gc_df, how='left', left_on='Target', right_on='node_num')
            cell_input_drugB_num_target_effect_att_df = cell_input_drugB_num_target_effect_att_df.sort_values(by=['Score'], ascending=False)
            # import pdb; pdb.set_trace()
            if whole_net:
                cell_input_drug_effect_test_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_pan/cell' + str(cell) + '_whole_input.csv', index=False, header=True)
                cell_input_drugA_num_target_effect_att_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_pan/cell' + str(cell) + '_drugA_whole_input.csv', index=False, header=True)
                cell_input_drugB_num_target_effect_att_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_pan/cell' + str(cell) + '_drugB_whole_input.csv', index=False, header=True)
            else:
                cell_input_drug_effect_test_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_pan/cell' + str(cell) + '_input.csv', index=False, header=True)
                cell_input_drugA_num_target_effect_att_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_pan/cell' + str(cell) + '_drugA_input.csv', index=False, header=True)
                cell_input_drugB_num_target_effect_att_df.to_csv('./analysis-' + dataname + '/fold_' + str(n_fold) + '_pan/cell' + str(cell) + '_drugB_input.csv', index=False, header=True)
            # import pdb; pdb.set_trace()

whole_net = False
n_fold = 1
PanGeneAnalyse().pan_drug_target(k=5, n_fold=n_fold, whole_net=whole_net, dataset=dataset, dataname=dataname)
