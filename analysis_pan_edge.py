import os
import pdb
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from geo_tmain_m3netflow import arg_parse, build_geotsgnn_model

class PanEdgeAnalyse():
    def __init__(self):
        pass

    def reform_weight_adj(self, fold_n, model, dataname):
        print('\nLOADING WEIGHT PARAMETERS FROM SAVED MODEL...')
        # COLLECT WEIGHT
        first_conv_up_weight = model.global_gnn.webconv_first.up_gene_edge_weight.cpu().data.numpy()
        first_conv_down_weight = model.global_gnn.webconv_first.down_gene_edge_weight.cpu().data.numpy()
        block_conv_up_weight = model.global_gnn.webconv_block.up_gene_edge_weight.cpu().data.numpy()
        block_conv_down_weight = model.global_gnn.webconv_block.down_gene_edge_weight.cpu().data.numpy()
        last_conv_up_weight = model.global_gnn.webconv_last.up_gene_edge_weight.cpu().data.numpy()
        last_conv_down_weight = model.global_gnn.webconv_last.down_gene_edge_weight.cpu().data.numpy()
        if os.path.exists('./analysis-' + dataname + '/fold_' + str(fold_n) + '_pan') == False:
            os.mkdir('./analysis-' + dataname + '/fold_' + str(fold_n) + '_pan')
        np.save('./analysis-' + dataname + '/fold_' + str(fold_n) + '_pan/first_conv_up_weight.npy', first_conv_up_weight)
        np.save('./analysis-' + dataname + '/fold_' + str(fold_n) + '_pan/first_conv_down_weight.npy', first_conv_down_weight)
        np.save('./analysis-' + dataname + '/fold_' + str(fold_n) + '_pan/block_conv_up_weight.npy', block_conv_up_weight)
        np.save('./analysis-' + dataname + '/fold_' + str(fold_n) + '_pan/block_conv_down_weight.npy', block_conv_down_weight)
        np.save('./analysis-' + dataname + '/fold_' + str(fold_n) + '_pan/last_conv_up_weight.npy', last_conv_up_weight)
        np.save('./analysis-' + dataname + '/fold_' + str(fold_n) + '_pan/last_conv_down_weight.npy', last_conv_down_weight)

        # MAKE ABSOLUTE VALUE
        first_conv_up_weight = np.absolute(first_conv_up_weight)
        first_conv_down_weight = np.absolute(first_conv_down_weight)
        block_conv_up_weight = np.absolute(block_conv_up_weight)
        block_conv_down_weight = np.absolute(block_conv_down_weight)
        last_conv_up_weight = np.absolute(last_conv_up_weight)
        last_conv_down_weight = np.absolute(last_conv_down_weight)
        conv_up_weight = (1/3) * (first_conv_up_weight + block_conv_up_weight + last_conv_up_weight)
        conv_down_weight = (1/3) * (first_conv_down_weight + block_conv_down_weight + last_conv_down_weight)
        conv_bind_weight = conv_up_weight + conv_down_weight

        # COMBINE WITH [kegg_gene_interaction.csv]
        kegg_gene_interaction_df = pd.read_csv('./' + dataset + '/filtered_data/kegg_gene_interaction.csv')
        kegg_gene_interaction_df['conv_bind_weight'] = conv_bind_weight

        node_num_dict_df = pd.read_csv('./analysis-' + dataname + '/node_num_dict.csv')
        node_num_dict = dict(zip(node_num_dict_df.node_name, node_num_dict_df.node_num))
        kegg_gene_interaction_df = kegg_gene_interaction_df.replace({'src': node_num_dict, 'dest': node_num_dict})
        kegg_gene_interaction_df.to_csv('./analysis-' + dataname + '/fold_' + str(fold_n) + '_pan/kegg_weighted_gene_interaction.csv', index=False, header=True)


if __name__ == "__main__":
    ##### REBUILD MODEL AND ANALYSIS PARAMTERS
    prog_args = arg_parse()
    device = torch.device('cuda:0') 
    model = build_geotsgnn_model(prog_args, device)

    ### DATASET SELECTION
    dataset = 'datainfo-nci'
    # dataset = 'datainfo-oneil'

    ### MODEL SELECTION
    modelname = 'tsgnn'
    # modelname = 'gat'
    # modelname = 'gcn'

    # SET THE FOLD FOR MODEL
    fold_n = 1
    if fold_n == 1:
        load_path = './' + dataset + '/result/' + modelname + '/epoch_100/best_train_model.pt'
    else:
        load_path = './' + dataset + '/result/' + modelname + '/epoch_100_' + str(fold_n - 1) + '/best_train_model.pt'
    model.load_state_dict(torch.load(load_path, map_location=device))
    PanEdgeAnalyse().reform_weight_adj(fold_n, model)
