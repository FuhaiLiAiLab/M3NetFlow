import os
import pdb
import torch
import argparse
import tensorboardX
import numpy as np
import pandas as pd
import torch.nn as nn
from scipy import sparse
from torch.autograd import Variable

import utils
from geo_loader.read_geograph import read_batch
from geo_loader.geograph_sampler import GeoGraphLoader
from enc_dec.geo_analysis_tsgnn_decoder import TSGNNDecoder

# PARSE ARGUMENTS FROM COMMAND LINE
def arg_parse():
    parser = argparse.ArgumentParser(description='COEMBED ARGUMENTS.')
    # ADD FOLLOWING ARGUMENTS
    parser.add_argument('--cuda', dest = 'cuda',
                help = 'CUDA.')
    parser.add_argument('--parallel', dest = 'parallel',
                help = 'Parrallel Computing')
    parser.add_argument('--GPU IDs', dest = 'gpu_ids',
                help = 'GPU IDs')
    parser.add_argument('--add-self', dest = 'adj_self',
                help = 'Graph convolution add nodes themselves.')
    parser.add_argument('--model', dest = 'model',
                help = 'Model load.')
    parser.add_argument('--lr', dest = 'lr', type = float,
                help = 'Learning rate.')
    parser.add_argument('--batch-size', dest = 'batch_size', type = int,
                help = 'Batch size.')
    parser.add_argument('--num_workers', dest = 'num_workers', type = int,
                help = 'Number of workers to load data.')
    parser.add_argument('--epochs', dest = 'num_epochs', type = int,
                help = 'Number of epochs to train.')
    parser.add_argument('--input-dim', dest = 'input_dim', type = int,
                help = 'Input feature dimension')
    parser.add_argument('--hidden-dim', dest = 'hidden_dim', type = int,
                help = 'Hidden dimension')
    parser.add_argument('--output-dim', dest = 'output_dim', type = int,
                help = 'Output dimension')
    parser.add_argument('--num-gc-layers', dest = 'num_gc_layers', type = int,
                help = 'Number of graph convolution layers before each pooling')
    parser.add_argument('--dropout', dest = 'dropout', type = float,
                help = 'Dropout rate.')

    # SET DEFAULT INPUT ARGUMENT
    parser.set_defaults(cuda = '0',
                        parallel = False,
                        add_self = '0', # 'add'
                        model = '0', # 'load'
                        lr = 0.002,
                        clip = 2.0,
                        batch_size = 16,
                        num_workers = 1,
                        num_epochs = 100,
                        input_dim = 8,
                        hidden_dim = 8,
                        output_dim = 24,
                        decoder_dim = 150,
                        dropout = 0.01)
    return parser.parse_args()

def build_geotsgnn_model(args, device, dataset):
    print('--- BUILDING UP TSGNN MODEL ... ---')
    # GET PARAMETERS
    # [num_gene, num_drug, (adj)node_num]
    final_annotation_gene_df = pd.read_csv('./' + dataset + '/filtered_data/kegg_gene_annotation.csv')
    gene_name_list = list(final_annotation_gene_df['kegg_gene'])
    num_gene = len(gene_name_list)
    drug_num_dict_df = pd.read_csv('./' + dataset + '/filtered_data/drug_num_dict.csv')
    drug_dict = dict(zip(drug_num_dict_df.Drug, drug_num_dict_df.drug_num))
    num_drug = len(drug_dict)
    node_num = num_gene + num_drug
    # [num_gene_edge, num_drug_edge]
    gene_num_df = pd.read_csv('./' + dataset + '/filtered_data/kegg_gene_num_interaction.csv')
    gene_num_df = gene_num_df.drop_duplicates()
    drugbank_num_df = pd.read_csv('./' + dataset + '/filtered_data/final_drugbank_num_sym.csv')
    num_gene_edge = gene_num_df.shape[0]
    num_drug_edge = drugbank_num_df.shape[0]
    num_edge = num_gene_edge + num_drug_edge
    # import pdb; pdb.set_trace()
    # BUILD UP MODEL
    model = TSGNNDecoder(input_dim=args.input_dim, hidden_dim=args.hidden_dim, embedding_dim=args.output_dim, decoder_dim=args.decoder_dim,
                num_gene=num_gene, node_num=node_num, num_edge=num_edge, num_gene_edge=num_gene_edge, device=device, dataset=dataset)
    model = model.to(device)
    return model


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
    

def analysis_geotsgnn_model(dataset_loader, adj, batch_random_final_dl_input_df, analysis_save_path, model, device, args):
    batch_loss = 0
    for batch_idx, data in enumerate(dataset_loader):
        x = Variable(data.x, requires_grad=False).to(device)
        edge_index = Variable(data.edge_index, requires_grad=False).to(device)
        drug_index = Variable(data.drug_index, requires_grad=False).to(device)
        label = Variable(data.label, requires_grad=True).to(device)
        adj = Variable(adj.float(), requires_grad=False).to(device)
        # THIS WILL USE METHOD [def forward()] TO MAKE PREDICTION
        ypred = model(x, edge_index, drug_index, adj, batch_random_final_dl_input_df, analysis_save_path)
        loss = model.loss(ypred, label)
        batch_loss += loss.item()
    return model, batch_loss, ypred


def analysis_geotsgnn(args, fold_n, model, analysis_save_path, device, dataset):
    print('-------------------------- ANALYSIS START --------------------------')
    print('-------------------------- ANALYSIS START --------------------------')
    print('-------------------------- ANALYSIS START --------------------------')
    print('-------------------------- ANALYSIS START --------------------------')
    print('-------------------------- ANALYSIS START --------------------------')
    # ANALYSIS MODEL ON WHOLE DATASET
    form_data_path = './' + dataset + '/form_analysis_data'
    xAll = np.load(form_data_path + '/xAll.npy')
    yAll = np.load(form_data_path + '/yAll.npy')
    drugAll =  np.load(form_data_path + '/drugAll.npy')
    random_final_dl_input_df = pd.read_csv('./' + dataset + '/filtered_data/random_final_dl_input.csv')
    # READ [adj, edge_index] FILES 
    adj = sparse_mx_to_torch_sparse_tensor(sparse.load_npz(form_data_path + '/adj_sparse.npz'))
    edge_index = torch.from_numpy(np.load(form_data_path + '/edge_index.npy') ).long() 

    dl_input_num = xAll.shape[0]
    batch_size = args.batch_size
    # CLEAN RESULT PREVIOUS EPOCH_I_PRED FILES
    # [num_feature, num_gene, num_drug]
    num_feature = 8
    dict_drug_num = pd.read_csv('./' + dataset + '/filtered_data/drug_num_dict.csv')
    num_drug = dict_drug_num.shape[0]
    final_annotation_gene_df = pd.read_csv('./' + dataset + '/filtered_data/kegg_gene_annotation.csv')
    num_gene = final_annotation_gene_df.shape[0]
    # [num_cellline]
    cell_line_list = sorted(list(set(random_final_dl_input_df['Cell Line Name'])))
    cell_line_num = [x for x in range(1, len(cell_line_list)+1)]
    cell_line_map_df = pd.DataFrame({'Cell_Line_Name': cell_line_list, 'Cell_Line_Num': cell_line_num})
    cell_line_map_df.to_csv('./' + dataset + '/filtered_data/cell_line_map_dict.csv', index=False, header=True)
    batch_included_cell_line_list = []
    # RUN ANALYSIS MODEL
    model.eval()
    all_ypred = np.zeros((1, 1))
    upper_index = 0
    batch_loss_list = []
    for index in range(0, dl_input_num, batch_size):
        if (index + batch_size) < dl_input_num:
            upper_index = index + batch_size
        else:
            upper_index = dl_input_num
        geo_datalist = read_batch(index, upper_index, xAll, yAll, drugAll,\
                num_feature, num_gene, num_drug, edge_index, dataset)
        dataset_loader, node_num, feature_dim = GeoGraphLoader.load_graph(geo_datalist, prog_args)
        batch_random_final_dl_input_df = random_final_dl_input_df.iloc[index : upper_index]
        print('ANALYZE MODEL...')
        # import pdb; pdb.set_trace()
        model, batch_loss, batch_ypred = analysis_geotsgnn_model(dataset_loader, adj, 
                                            batch_random_final_dl_input_df, analysis_save_path, model, device, args)
        print('BATCH LOSS: ', batch_loss)
        batch_loss_list.append(batch_loss)
        # PRESERVE PREDICTION OF BATCH TEST DATA
        batch_ypred = (Variable(batch_ypred).data).cpu().numpy()
        all_ypred = np.vstack((all_ypred, batch_ypred))
        # TIME TO STOP SINCE ALL [cell line] WERE INCLUDED
        tmp_batch_cell_line_list = sorted(list(set(batch_random_final_dl_input_df['Cell Line Name'])))
        batch_included_cell_line_list += tmp_batch_cell_line_list
        batch_included_cell_line_list = sorted(list(set(batch_included_cell_line_list)))
        # import pdb; pdb.set_trace()
        if batch_included_cell_line_list == cell_line_list:
            print(len(batch_included_cell_line_list))
            print(batch_included_cell_line_list)
            break



if __name__ == "__main__":
    # PARSE ARGUMENT FROM TERMINAL OR DEFAULT PARAMETERS
    prog_args = arg_parse()

    # CHECK AND ALLOCATE RESOURCES
    device, prog_args.gpu_ids = utils.get_available_devices()
    # MANUAL SET
    device = torch.device('cuda:0') 
    torch.cuda.set_device(device)
    print('MAIN DEVICE: ', device)
    # SINGLE GPU
    prog_args.gpu_ids = [0]
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    ### DATASET SELECTION
    dataset = 'datainfo-nci'
    dataname = 'nci'
    # dataset = 'datainfo-oneil'
    # dataset = 'oneil

    ### MODEL SELECTION
    modelname = 'tsgnn'
    # modelname = 'gat'
    # modelname = 'gcn'
    
    while os.path.exists('./analysis-' + dataname) == False:
        os.mkdir('./analysis-' + dataname)

    # # ANALYSIS THE MODEL
    # ANALYSIS USING [FOLD-1]
    fold_n = 5
    while os.path.exists('./analysis-' + dataname + '/fold_' + str(fold_n)) == False:
        os.mkdir('./analysis-' + dataname + '/fold_' + str(fold_n))
    model = build_geotsgnn_model(prog_args, device, dataset)
    if fold_n == 1:
        analysis_load_path = './' + dataset + '/result/' + modelname + '/epoch_100/best_train_model.pt'
    else:
        analysis_load_path = './' + dataset + '/result/' + modelname + '/epoch_100_' + str(fold_n - 1) + '/best_train_model.pt'
    analysis_save_path = './analysis-' + dataname + '/fold_' + str(fold_n)
    model.load_state_dict(torch.load(analysis_load_path, map_location=device))
    analysis_geotsgnn(prog_args, fold_n, model, analysis_save_path, device, dataset)