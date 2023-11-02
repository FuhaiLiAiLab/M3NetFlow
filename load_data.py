import os
import torch
import numpy as np
import pandas as pd

from numpy import inf
from scipy import sparse
from sklearn.model_selection import train_test_split

class LoadData():
    def __init__(self):
        pass

    def load_batch(self, index, upper_index, place_num, dataset, drug_feature=False):
        # PRELOAD EACH SPLIT DATASET
        split_input_df = pd.read_csv('./' + dataset + '/filtered_data/split_input_' + str(place_num + 1) + '.csv')
        num_feature = 8
        final_annotation_gene_df = pd.read_csv('./' + dataset + '/filtered_data/kegg_gene_annotation.csv')
        gene_name_list = list(final_annotation_gene_df['kegg_gene'])
        print('READING GENE FEATURES FILES ...')
        final_gdsc_rna_df = pd.read_csv('./' + dataset + '/filtered_data/final_rna.csv')
        final_cmeth_max_df = pd.read_csv('./' + dataset + '/filtered_data/final_cmeth_max.csv')
        final_cmeth_min_df = pd.read_csv('./' + dataset + '/filtered_data/final_cmeth_min.csv')
        final_gdsc_cnv_df = pd.read_csv('./' + dataset + '/filtered_data/final_cnv.csv')
        final_cmut_amp_df = pd.read_csv('./' + dataset + '/filtered_data/final_cmut_amp.csv')
        final_cmut_del_df = pd.read_csv('./' + dataset + '/filtered_data/final_cmut_del.csv')
        num_gene, num_cellline = final_gdsc_rna_df.shape
        # CONVERT [drugbank.csv] TO A LIST
        print('READING DRUGBANK ...')
        final_drugbank_df = pd.read_csv('./' + dataset + '/filtered_data/final_drugbank.csv')
        final_drugbank_comlist = final_drugbank_df.values.tolist()
        print('READING DRUGDICT ...')
        drug_num_dict_df = pd.read_csv('./' + dataset + '/filtered_data/drug_num_dict.csv')
        drug_dict = dict(zip(drug_num_dict_df.Drug, drug_num_dict_df.drug_num))
        num_drug = len(drug_dict)
        print('READING FINISHED ...')
        # COMBINE A BATCH SIZE AS [x_batch, y_batch, drug_batch]
        print('-----' + str(index) + ' to ' + str(upper_index) + '-----')
        tmp_batch_size = 0
        y_input_list = []
        drug_input_list = []
        x_batch = np.zeros((1, (num_feature * (num_gene + num_drug))))
        for row in split_input_df.iloc[index : upper_index].itertuples():
            tmp_batch_size += 1
            drug_a = row[1]
            drug_b = row[2]
            cellline_name = row[3]
            y = row[4]
            # DRUG_A AND [4853] TARGET GENES
            one_drug_target_list = []
            duo_drug_target_list = []
            for gene in gene_name_list:
                drugA_target = [drug_a, gene] in final_drugbank_comlist
                drugB_target = [drug_b, gene] in final_drugbank_comlist
                if drugA_target and drugB_target:
                    one_drug_target_list.append(0.0)
                    duo_drug_target_list.append(1.0)
                elif drugA_target or drugB_target:
                    one_drug_target_list.append(1.0)
                    duo_drug_target_list.append(0.0)
                else:
                    one_drug_target_list.append(0.0)
                    duo_drug_target_list.append(0.0)
            # GENE FEATURES SEQUENCE
            gene_rna_list = [float(x) for x in list(final_gdsc_rna_df[cellline_name])]
            gene_cmeth_max_list = [float(x) for x in list(final_cmeth_max_df[cellline_name])]
            gene_cmeth_min_list = [float(x) for x in list(final_cmeth_min_df[cellline_name])]
            gene_cnv_list = [float(x) for x in list(final_gdsc_cnv_df[cellline_name])]
            gene_cmut_amp_list = [float(x) for x in list(final_cmut_amp_df[cellline_name])]
            gene_cmut_del_list = [float(x) for x in list(final_cmut_del_df[cellline_name])]
            # COMBINE [drugA, drugB, rna, cmeth] 
            x_input_list = []
            for i in range(num_gene):
                # APPEND DRUG INFORMATION
                x_input_list.append(one_drug_target_list[i])
                x_input_list.append(duo_drug_target_list[i])
                # APPEND GENE FEATURES
                x_input_list.append(gene_rna_list[i])
                x_input_list.append(gene_cmeth_max_list[i])
                x_input_list.append(gene_cmeth_min_list[i])
                x_input_list.append(gene_cnv_list[i])
                x_input_list.append(gene_cmut_amp_list[i])
                x_input_list.append(gene_cmut_del_list[i])
            if drug_feature == False:
                fillin_list = [0.0] * num_feature
                for i in range(num_drug):
                    x_input_list += fillin_list
            x_input = np.array(x_input_list)
            x_batch = np.vstack((x_batch, x_input))
            # COMBINE DRUG[A/B] LIST
            drug_input_list.append(drug_dict[drug_a])
            drug_input_list.append(drug_dict[drug_b])
            # COMBINE SCORE LIST
            y_input_list.append(y)
        # import pdb; pdb.set_trace()
        x_batch = np.delete(x_batch, 0, axis = 0)
        y_batch = np.array(y_input_list).reshape(tmp_batch_size, 1)
        drug_batch = np.array(drug_input_list).reshape(tmp_batch_size, 2)
        print(x_batch.shape)
        print(y_batch.shape)
        print(drug_batch.shape)
        return x_batch, y_batch, drug_batch


    def load_all_split(self, batch_size, k, dataset):
        form_data_path = './' + dataset + '/form_data'
        # LOAD 100 PERCENT DATA
        print('LOADING ALL SPLIT DATA...')
        # FIRST LOAD EACH SPLIT DATA
        for place_num in range(k):
            split_input_df = pd.read_csv('./' + dataset + '/filtered_data/split_input_' + str(place_num + 1) + '.csv')
            input_num, input_dim = split_input_df.shape
            num_feature = 8
            final_annotation_gene_df = pd.read_csv('./' + dataset + '/filtered_data/kegg_gene_annotation.csv')
            gene_name_list = list(final_annotation_gene_df['kegg_gene'])
            num_gene = len(gene_name_list)
            drug_num_dict_df = pd.read_csv('./' + dataset + '/filtered_data/drug_num_dict.csv')
            drug_dict = dict(zip(drug_num_dict_df.Drug, drug_num_dict_df.drug_num))
            num_drug = len(drug_dict)
            x_split = np.zeros((1, num_feature * (num_gene + num_drug)))
            y_split = np.zeros((1, 1))
            drug_split = np.zeros((1, 2))
            for index in range(0, input_num, batch_size):
                if (index + batch_size) < input_num:
                    upper_index = index + batch_size
                else:
                    upper_index = input_num
                x_batch, y_batch, drug_batch = LoadData().load_batch(index, upper_index, place_num, dataset)
                x_split = np.vstack((x_split, x_batch))
                y_split = np.vstack((y_split, y_batch))
                drug_split = np.vstack((drug_split, drug_batch))
            x_split = np.delete(x_split, 0, axis = 0)
            y_split = np.delete(y_split, 0, axis = 0)
            drug_split = np.delete(drug_split, 0, axis = 0)
            print('-------SPLIT DATA SHAPE-------')
            print(x_split.shape)
            print(y_split.shape)
            print(drug_split.shape)
            np.save(form_data_path + '/x_split' + str(place_num + 1) + '.npy', x_split)
            np.save(form_data_path + '/y_split' + str(place_num + 1) + '.npy', y_split)
            np.save(form_data_path + '/drug_split' + str(place_num + 1) + '.npy', drug_split)
            

    def load_train_test(self, k, n_fold, dataset):
        form_data_path = './' + dataset + '/form_data'
        num_feature = 8
        final_annotation_gene_df = pd.read_csv('./' + dataset + '/filtered_data/kegg_gene_annotation.csv')
        gene_name_list = list(final_annotation_gene_df['kegg_gene'])
        num_gene = len(gene_name_list)
        drug_num_dict_df = pd.read_csv('./' + dataset + '/filtered_data/drug_num_dict.csv')
        drug_dict = dict(zip(drug_num_dict_df.Drug, drug_num_dict_df.drug_num))
        num_drug = len(drug_dict)
        xTr = np.zeros((1, num_feature * (num_gene + num_drug)))
        yTr = np.zeros((1, 1))
        drugTr = np.zeros((1, 2))
        for i in range(1, k + 1):
            if i == n_fold:
                print('--- LOADING ' + str(i) + '-TH SPLIT TEST DATA ---')
                xTe = np.load(form_data_path + '/x_split' + str(i) + '.npy')
                yTe = np.load(form_data_path + '/y_split' + str(i) + '.npy')
                drugTe = np.load(form_data_path + '/drug_split' + str(i) + '.npy')
            else:
                print('--- LOADING ' + str(i) + '-TH SPLIT TRAINING DATA ---')
                x_split = np.load(form_data_path + '/x_split' + str(i) + '.npy')
                y_split = np.load(form_data_path + '/y_split' + str(i) + '.npy')
                drug_split = np.load(form_data_path + '/drug_split' + str(i) + '.npy')
                print('--- COMBINING DATA ... ---')
                xTr = np.vstack((xTr, x_split))
                yTr = np.vstack((yTr, y_split))
                drugTr = np.vstack((drugTr, drug_split))
        print('--- TRAINING INPUT SHAPE ---')
        xTr = np.delete(xTr, 0, axis = 0)
        yTr = np.delete(yTr, 0, axis = 0)
        drugTr = np.delete(drugTr, 0, axis = 0)
        print(xTr.shape)
        print(yTr.shape)
        print(drugTr.shape)
        np.save(form_data_path + '/xTr' + str(n_fold) + '.npy', xTr)
        np.save(form_data_path + '/yTr' + str(n_fold) + '.npy', yTr)
        np.save(form_data_path + '/drugTr' + str(n_fold) + '.npy', drugTr)
        print('--- TEST INPUT SHAPE ---')
        print(xTe.shape)
        print(yTe.shape)
        print(drugTe.shape)
        np.save(form_data_path + '/xTe' + str(n_fold) + '.npy', xTe)
        np.save(form_data_path + '/yTe' + str(n_fold) + '.npy', yTe)
        np.save(form_data_path + '/drugTe' + str(n_fold) + '.npy', drugTe)

    def combine_whole_dataset(self, k, dataset):
        form_data_path = './' + dataset + '/form_data'
        num_feature = 8
        final_annotation_gene_df = pd.read_csv('./' + dataset + '/filtered_data/kegg_gene_annotation.csv')
        gene_name_list = list(final_annotation_gene_df['kegg_gene'])
        num_gene = len(gene_name_list)
        drug_num_dict_df = pd.read_csv('./' + dataset + '/filtered_data/drug_num_dict.csv')
        drug_dict = dict(zip(drug_num_dict_df.Drug, drug_num_dict_df.drug_num))
        num_drug = len(drug_dict)
        xAll = np.zeros((1, num_feature * (num_gene + num_drug)))
        yAll = np.zeros((1, 1))
        drugAll = np.zeros((1, 2))
        for i in range(1, k + 1):
            print('--- LOADING ' + str(i) + '-TH SPLIT TRAINING DATA ---')
            x_split = np.load(form_data_path + '/x_split' + str(i) + '.npy')
            y_split = np.load(form_data_path + '/y_split' + str(i) + '.npy')
            drug_split = np.load(form_data_path + '/drug_split' + str(i) + '.npy')
            print('--- COMBINING DATA ... ---')
            xAll = np.vstack((xAll, x_split))
            yAll = np.vstack((yAll, y_split))
            drugAll = np.vstack((drugAll, drug_split))
        print('--- ALL DATASET INPUT SHAPE ---')
        xAll = np.delete(xAll, 0, axis = 0)
        yAll = np.delete(yAll, 0, axis = 0)
        drugAll = np.delete(drugAll, 0, axis = 0)
        print(xAll.shape)
        print(yAll.shape)
        print(drugAll.shape)
        np.save(form_data_path + '/xAll.npy', xAll)
        np.save(form_data_path + '/yAll.npy', yAll)
        np.save(form_data_path + '/drugAll.npy', drugAll)
    
    def load_adj_edgeindex(self, dataset):
        form_data_path = './' + dataset + '/form_data'
        # FORM A WHOLE ADJACENT MATRIX
        gene_num_df = pd.read_csv('./' + dataset + '/filtered_data/kegg_gene_num_interaction.csv')
        src_gene_list = list(gene_num_df['src'])
        dest_gene_list = list(gene_num_df['dest'])
        final_annotation_gene_df = pd.read_csv('./' + dataset + '/filtered_data/kegg_gene_annotation.csv')
        gene_name_list = list(final_annotation_gene_df['kegg_gene'])
        num_gene = len(gene_name_list)
        dict_drug_num = pd.read_csv('./' + dataset + '/filtered_data/drug_num_dict.csv')
        num_drug = dict_drug_num.shape[0]
        num_node = num_gene + num_drug
        adj = np.zeros((num_node, num_node))
        # GENE-GENE ADJACENT MATRIX
        for i in range(len(src_gene_list)):
            row_idx = src_gene_list[i] - 1
            col_idx = dest_gene_list[i] - 1
            adj[row_idx, col_idx] = 1
            adj[col_idx, row_idx] = 1 # WHETHER WE WANT ['sym']
        # import pdb; pdb.set_trace()
        # DRUG_TARGET ADJACENT MATRIX
        drugbank_num_df = pd.read_csv('./' + dataset + '/filtered_data/final_drugbank_num.csv')
        drugbank_drug_list = list(drugbank_num_df['Drug'])
        drugbank_target_list = list(drugbank_num_df['Target'])
        for row in drugbank_num_df.itertuples():
            row_idx = row[1] - 1
            col_idx = row[2] - 1
            adj[row_idx, col_idx] = 1
            adj[col_idx, row_idx] = 1
        # import pdb; pdb.set_trace()
        # np.save(form_data_path + '/adj.npy', adj)
        adj_sparse = sparse.csr_matrix(adj)
        sparse.save_npz(form_data_path + '/adj_sparse.npz', adj_sparse)
        # [edge_index]
        genedrug_src_list = src_gene_list + drugbank_drug_list + drugbank_target_list
        genedrug_dest_list = dest_gene_list + drugbank_target_list + drugbank_drug_list
        genedrug_src_indexlist = []
        genedrug_dest_indexlist = []
        for i in range(len(genedrug_src_list)):
            genedrug_src_index = genedrug_src_list[i] - 1
            genedrug_src_indexlist.append(genedrug_src_index)
            genedrug_dest_index = genedrug_dest_list[i] - 1
            genedrug_dest_indexlist.append(genedrug_dest_index)
        edge_index = np.column_stack((genedrug_src_indexlist, genedrug_dest_indexlist)).T
        np.save(form_data_path + '/edge_index.npy', edge_index)

    def load_edge_adj(self, dataset):
        form_data_path = './' + dataset + '/form_data'
        # FORM A WHOLE ADJACENT MATRIX
        gene_num_df = pd.read_csv('./' + dataset + '/filtered_data/kegg_gene_num_interaction.csv')
        gene_num_df = gene_num_df.drop_duplicates()
        final_annotation_gene_df = pd.read_csv('./' + dataset + '/filtered_data/kegg_gene_annotation.csv')
        gene_name_list = list(final_annotation_gene_df['kegg_gene'])
        num_gene = len(gene_name_list)
        dict_drug_num = pd.read_csv('./' + dataset + '/filtered_data/drug_num_dict.csv')
        num_drug = dict_drug_num.shape[0]
        num_node = num_gene + num_drug
        # DRUG_TARGET ADJACENT MATRIX
        drugbank_num_df = pd.read_csv('./' + dataset + '/filtered_data/final_drugbank_num_sym.csv')
        drugbank_num_df = drugbank_num_df.rename(columns={'Drug': 'src', 'Target': 'dest'})
        # CONCAT TWO DATAFRAMES
        network_num_df = pd.concat([gene_num_df, drugbank_num_df], ignore_index=True)
        num_edge = network_num_df.shape[0]
        # BUILD UP [trans]
        trans = np.zeros((num_node, num_edge))
        for row in network_num_df.itertuples():
            edge_idx = row[0]
            node_src_idx = row[1] - 1
            node_dest_idx = row[2] - 1
            trans[node_src_idx, edge_idx] = 1
            trans[node_dest_idx, edge_idx] = 1
        # np.save(form_data_path + '/trans.npy', trans)
        trans_sparse = sparse.csr_matrix(trans)
        sparse.save_npz(form_data_path + '/trans_sparse.npz', trans_sparse)
        # BUILD UP [edge_adj]
        trans_1 = trans_sparse
        trans_2 = trans_sparse.transpose()
        edge_adj = sparse.csr_matrix.dot(trans_2, trans_1)
        edge_adj = edge_adj.todense()
        np.fill_diagonal(edge_adj, 0)
        edge_adj[edge_adj > 1] = 1
        # import pdb; pdb.set_trace()
        # np.save(form_data_path + '/edge_adj.npy', edge_adj)
        edge_adj_sparse = sparse.csr_matrix(edge_adj)
        sparse.save_npz(form_data_path + '/edge_adj_sparse.npz', edge_adj_sparse)
        # BUILD EDGE FEATURE [edge_feat]
        edge_conn = np.sum(edge_adj, axis=1).reshape(-1,1) # EDGE CONNECTION FEATURE
        edge_feat = np.zeros((num_edge, 2))
        for row in network_num_df.itertuples():
            edge_idx = row[0]
            if edge_idx >= gene_num_df.shape[0]:
                edge_feat[edge_idx, 1] = 1 # EDGE [Drug-Gene]
            else:
                edge_feat[edge_idx, 0] = 1 # EDGE [Gene-Gene]
        edge_feat = np.hstack((edge_feat, edge_conn))
        np.save(form_data_path + '/edge_feat.npy', edge_feat)

    def load_path_adj_edgeindex(self, dataset):
        form_data_path = './' + dataset + '/form_data'
        kegg_path_gene_interaction_df = pd.read_csv('./' + dataset + '/filtered_data/kegg_path_gene_interaction.csv')
        # FORM NOTATION TO [signaling pathways]
        kegg_sp_list = list(set(kegg_path_gene_interaction_df['path']))
        kegg_sp_list.sort()
        kegg_sp_notation_list = ['sp' + str(x) for x in range(1, len(kegg_sp_list)+1)]
        kegg_sp_map_df = pd.DataFrame({'SignalingPath': kegg_sp_list, 'SpNotation': kegg_sp_notation_list})
        kegg_sp_map_df.to_csv('./' + dataset + '/filtered_data/kegg_sp_map.csv', index=False, header=True)
        kegg_sp_map_dict = dict(zip(kegg_sp_list, kegg_sp_notation_list))
        # REPLACE [gene_num, signalingpath] TO [gene_num, sp]
        kegg_gene_num_dict_df = pd.read_csv('./' + dataset + '/filtered_data/kegg_gene_num_dict.csv')
        kegg_gene_num_dict = dict(zip(kegg_gene_num_dict_df.kegg_gene, kegg_gene_num_dict_df.gene_num))
        kegg_path_gene_interaction_df = kegg_path_gene_interaction_df.replace({'src': kegg_gene_num_dict, 
                                                                               'dest': kegg_gene_num_dict,
                                                                               'path': kegg_sp_map_dict})
        kegg_gene_num_dict_df = pd.read_csv('./' + dataset + '/filtered_data/kegg_gene_num_dict.csv')
        gene_num = kegg_gene_num_dict_df.shape[0]
        for num_sp in kegg_sp_notation_list:
            zero_adj = np.zeros((gene_num, gene_num))
            sp_kegg_path_gene_df = kegg_path_gene_interaction_df.loc[kegg_path_gene_interaction_df['path'] == num_sp]
            src_sp_gene_list = list(sp_kegg_path_gene_df['src'])
            dest_sp_gene_list = list(sp_kegg_path_gene_df['dest'])
            sp_gene_list = sorted(list(set(src_sp_gene_list + dest_sp_gene_list)))
            sp_gene_idx_array = np.array([x-1 for x in sp_gene_list])
            # import pdb; pdb.set_trace()
            # GENE-GENE ADJACENT MATRIX
            for i in range(len(src_sp_gene_list)):
                row_idx = src_sp_gene_list[i] - 1
                col_idx = dest_sp_gene_list[i] - 1
                zero_adj[row_idx, col_idx] = 1
                zero_adj[col_idx, row_idx] = 1 # WHETHER WE WANT ['sym']
            zero_adj[zero_adj > 1] = 1
            sp_adj = zero_adj[sp_gene_idx_array][:, sp_gene_idx_array]
            print(np.sum(sp_adj))
            np.save(form_data_path + '/' + num_sp + '_adj.npy', sp_adj)
            np.save(form_data_path + '/' + num_sp + '_gene_idx.npy', sp_gene_idx_array)
            # import pdb; pdb.set_trace()
    

    def load_path_adj_khop_mask_edgeindex(self, khop_num, dataset):
        form_data_path = './' + dataset + '/form_data'
        kegg_path_gene_interaction_df = pd.read_csv('./' + dataset + '/filtered_data/kegg_path_gene_interaction.csv')
        # FORM NOTATION TO [signaling pathways]
        kegg_sp_list = list(set(kegg_path_gene_interaction_df['path']))
        kegg_sp_list.sort()
        kegg_sp_notation_list = ['sp' + str(x) for x in range(1, len(kegg_sp_list)+1)]
        kegg_sp_map_df = pd.DataFrame({'SignalingPath': kegg_sp_list, 'SpNotation': kegg_sp_notation_list})
        kegg_sp_map_df.to_csv('./' + dataset + '/filtered_data/kegg_sp_map.csv', index=False, header=True)
        kegg_sp_map_dict = dict(zip(kegg_sp_list, kegg_sp_notation_list))
        # REPLACE [gene_num, signalingpath] TO [gene_num, sp]
        kegg_gene_num_dict_df = pd.read_csv('./' + dataset + '/filtered_data/kegg_gene_num_dict.csv')
        kegg_gene_num_dict = dict(zip(kegg_gene_num_dict_df.kegg_gene, kegg_gene_num_dict_df.gene_num))
        kegg_path_gene_interaction_df = kegg_path_gene_interaction_df.replace({'src': kegg_gene_num_dict, 
                                                                               'dest': kegg_gene_num_dict,
                                                                               'path': kegg_sp_map_dict})
        kegg_gene_num_dict_df = pd.read_csv('./' + dataset + '/filtered_data/kegg_gene_num_dict.csv')
        gene_num = kegg_gene_num_dict_df.shape[0]
        for num_sp in kegg_sp_notation_list:
            zero_adj = np.zeros((gene_num, gene_num))
            sp_kegg_path_gene_df = kegg_path_gene_interaction_df.loc[kegg_path_gene_interaction_df['path'] == num_sp]
            src_sp_gene_list = list(sp_kegg_path_gene_df['src'])
            dest_sp_gene_list = list(sp_kegg_path_gene_df['dest'])
            sp_gene_list = sorted(list(set(src_sp_gene_list + dest_sp_gene_list)))
            sp_gene_idx_array = np.array([x-1 for x in sp_gene_list])
            # MAP [sp_gene_list] TO [subgraph index / gene name]
            khop_subidx = np.arange(sp_gene_idx_array.shape[0] * 3)
            addone_sp_gene_idx_array = sp_gene_idx_array + 1
            khop_sp_gene_idx_array = np.tile(addone_sp_gene_idx_array, 3)
            sp_gene_map_df = pd.DataFrame({'Sub_idx': khop_subidx, 'Node_idx': khop_sp_gene_idx_array})
            sp_gene_map_df = pd.merge(sp_gene_map_df, kegg_gene_num_dict_df, how='left', left_on='Node_idx', right_on='gene_num')
            sp_gene_map_df.to_csv(form_data_path + '/' + num_sp + '_gene_map.csv', index=False, header=True)
            # GENE-GENE ADJACENT MATRIX
            for i in range(len(src_sp_gene_list)):
                row_idx = src_sp_gene_list[i] - 1
                col_idx = dest_sp_gene_list[i] - 1
                zero_adj[row_idx, col_idx] = 1
                zero_adj[col_idx, row_idx] = 1 # WHETHER WE WANT ['sym']
            zero_adj[zero_adj > 1] = 1
            sp_adj = zero_adj[sp_gene_idx_array][:, sp_gene_idx_array]
            print(np.sum(sp_adj))
            # CALL [khop_mask_edgeindex()]
            khop_pow_subadj_edgeindex, khop_mask_edgeindex = LoadData().khop_mask_edgeindex(sp_adj, khop_num)
            khop_pow_subadj_edgeindex = khop_pow_subadj_edgeindex.numpy()
            khop_mask_edgeindex = khop_mask_edgeindex.numpy()
            np.save(form_data_path + '/' + num_sp + '_khop_subadj_edgeindex.npy', khop_pow_subadj_edgeindex)
            np.save(form_data_path + '/' + num_sp + '_khop_mask_edgeindex.npy', khop_mask_edgeindex)
            np.save(form_data_path + '/' + num_sp + '_gene_idx.npy', sp_gene_idx_array)

    def khop_sum_subadj(self, subadj, khop_num):
        khop_sum_subadj = subadj.clone()
        for i in range(2, khop_num + 1):
            ith_pow_subadj = torch.matrix_power(subadj, i)
            khop_sum_subadj += ith_pow_subadj
        khop_sum_subadj[khop_sum_subadj > 0] = 1.0
        khop_sum_subadj = khop_sum_subadj - torch.eye(subadj.shape[0])
        khop_sum_subadj[khop_sum_subadj < 0] = 0.0
        return khop_sum_subadj

    def khop_mask_edgeindex(self, subadj, khop_num):
        subadj = torch.from_numpy(subadj)
        subadj_edgeindex = subadj.to_sparse()._indices()
        ### FORM [khop] [korder edgeindex] & [mask]
        # INITIAL [korder] MATRIX TO INCLUDE ALL EDGE FROM 1-HOP TO K-HOP
        korder_pow_subadj = LoadData().khop_sum_subadj(subadj, khop_num)
        korder_pow_subadj_edgeindex = korder_pow_subadj.to_sparse()._indices()
        ### FOR LOOP FOR [i-th hop] [korder edgeindex] & [mask]
        # [mask]
        mask_edgeindex = subadj[korder_pow_subadj_edgeindex[0,:], korder_pow_subadj_edgeindex[1,:]] # [1-hop mask]
        khop_mask_edgeindex = mask_edgeindex
        # [korder]
        tmp_korder_pow_subadj_edgeindex = korder_pow_subadj_edgeindex
        khop_pow_subadj_edgeindex = korder_pow_subadj_edgeindex
        # [acc_pow]
        acc_pow_subadj = torch.matrix_power(subadj, 1)
        for i in range(2, khop_num + 1):
            # CALCULATE THE [i-th hop] ADJACENT MATRIX
            tmp_pow_subadj = torch.matrix_power(subadj, i)
            tmp_pow_subadj[tmp_pow_subadj > 0] = 1.0
            tmp_pow_subadj = tmp_pow_subadj - acc_pow_subadj - torch.eye(subadj.shape[0])
            tmp_pow_subadj[tmp_pow_subadj < 0] = 0.0
            acc_pow_subadj += tmp_pow_subadj
            # CALCULATE THE [i-th hop] MASK INDEX
            tmp_mask_edgeindex = tmp_pow_subadj[korder_pow_subadj_edgeindex[0,:], korder_pow_subadj_edgeindex[1,:]] # [i-th hop mask]
            khop_mask_edgeindex = torch.cat([khop_mask_edgeindex, tmp_mask_edgeindex])
            # COMBINE EACH HOP [edge_index]
            tmp_korder_pow_subadj_edgeindex = tmp_korder_pow_subadj_edgeindex + subadj.shape[0]
            khop_pow_subadj_edgeindex = torch.cat([khop_pow_subadj_edgeindex, tmp_korder_pow_subadj_edgeindex], dim=1)
        return khop_pow_subadj_edgeindex, khop_mask_edgeindex


# DATASET SELECTION
dataset = 'datainfo-nci'
# dataset = 'datainfo-oneil'

# LoadData().load_path_adj_edgeindex(dataset=dataset)

# print('-----------')
# LoadData().load_path_adj_khop_mask_edgeindex(khop_num=3, dataset=dataset)
# LoadData().combine_whole_dataset(k=5, dataset=dataset)