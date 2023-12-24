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
from enc_dec.geo_gat_decoder import GATDecoder

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
                        batch_size = 64,
                        num_workers = 1,
                        num_epochs = 100,
                        input_dim = 8,
                        hidden_dim = 24,
                        output_dim = 24,
                        decoder_dim = 150,
                        dropout = 0.01)
    return parser.parse_args()


def learning_rate_schedule(args, dl_input_num, iteration_num, e1, e2, e3, e4):
    epoch_iteration = int(dl_input_num / args.batch_size)
    l1 = (args.lr - 0.0008) / (e1 * epoch_iteration)
    l2 = (0.0008 - 0.0006) / (e2 * epoch_iteration)
    l3 = (0.0006 - 0.0005) / (e3 * epoch_iteration)
    l4 = (0.0005 - 0.0001) / (e4 * epoch_iteration)
    l5 = 0.0001
    if iteration_num <= (e1 * epoch_iteration):
        learning_rate = args.lr - iteration_num * l1
    elif iteration_num <= (e1 + e2) * epoch_iteration:
        learning_rate = 0.0008 - (iteration_num - e1 * epoch_iteration) * l2
    elif iteration_num <= (e1 + e2 + e3) * epoch_iteration:
        learning_rate = 0.0006 - (iteration_num - (e1 + e2) * epoch_iteration) * l3
    elif iteration_num <= (e1 + e2 + e3 + e4) * epoch_iteration:
        learning_rate = 0.0005 - (iteration_num - (e1 + e2 + e3) * epoch_iteration) * l4
    else:
        learning_rate = l5
    print('-------LEARNING RATE: ' + str(learning_rate) + '-------' )
    return learning_rate


def build_geogat_model(args, device, dataset):
    print('--- BUILDING UP GAT MODEL ... ---')
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
    model = GATDecoder(input_dim=args.input_dim, hidden_dim=args.hidden_dim, embedding_dim=args.output_dim, 
                decoder_dim=args.decoder_dim, num_head=1, device=device)
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


def train_geogat_model(dataset_loader, model, device, args, learning_rate):
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=learning_rate, eps=1e-7, weight_decay=1e-15)
    batch_loss = 0
    for batch_idx, data in enumerate(dataset_loader):
        optimizer.zero_grad()
        # import pdb; pdb.set_trace()
        x = Variable(data.x.float(), requires_grad=False).to(device)
        edge_index = Variable(data.edge_index, requires_grad=False).to(device)
        drug_index = Variable(data.drug_index, requires_grad=False).to(device)
        label = Variable(data.label, requires_grad=False).to(device)
        ypred = model(x, edge_index, drug_index)
        loss = model.loss(ypred, label)
        loss.backward()
        batch_loss += loss.item()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
    torch.cuda.empty_cache()
    return model, batch_loss, ypred


def train_geogat(args, fold_n, load_path, iteration_num, device, dataset):
    # TRAINING DATASET BASIC PARAMETERS
    # [num_feature, num_gene, num_drug]
    num_feature = 8
    dict_drug_num = pd.read_csv('./' + dataset + '/filtered_data/drug_num_dict.csv')
    num_drug = dict_drug_num.shape[0]
    final_annotation_gene_df = pd.read_csv('./' + dataset + '/filtered_data/kegg_gene_annotation.csv')
    num_gene = final_annotation_gene_df.shape[0]
    dir_opt = '/' + dataset + ''
    form_data_path = '.' + dir_opt + '/form_data'
    # READ THESE FEATURE LABEL FILES
    print('--- LOADING TRAINING FILES ... ---')
    xTr = np.load('./' + dataset + '/form_data/xTr' + str(fold_n) + '.npy')
    yTr = np.load('./' + dataset + '/form_data/yTr' + str(fold_n) + '.npy')
    drugTr =  np.load('./' + dataset + '/form_data/drugTr' + str(fold_n) + '.npy')
    edge_index = torch.from_numpy(np.load(form_data_path + '/edge_index.npy') ).long() 

    # BUILD [WeightBiGNN, DECODER] MODEL
    model = build_geogat_model(args, device, dataset)
    if args.model == 'load':
        model.load_state_dict(torch.load(load_path, map_location=device))

    # TRAIN MODEL ON TRAINING DATASET
    # OTHER PARAMETERS
    dl_input_num = xTr.shape[0]
    epoch_num = args.num_epochs
    learning_rate = args.lr
    batch_size = args.batch_size
    # RECORD EPOCH LOSS AND PEARSON CORRELATION
    if args.model != 'load':
        iteration_num = 0
    max_test_corr = 0
    max_test_corr_id = 0
    e1 = 20
    e2 = 10
    e3 = 10
    e4 = 20
    epoch_loss_list = []
    epoch_pearson_list = []
    test_loss_list = []
    test_pearson_list = []
    # CLEAN RESULT PREVIOUS EPOCH_I_PRED FILES
    folder_name = 'epoch_' + str(epoch_num)
    path = '.' + dir_opt + '/result/%s' % (folder_name)
    unit = 1
    while os.path.exists('.' + dir_opt + '/result') == False:
        os.mkdir('.' + dir_opt + '/result')
    while os.path.exists(path):
        path = '.' + dir_opt + '/result/%s_%d' % (folder_name, unit)
        unit += 1
    os.mkdir(path)
    # import pdb; pdb.set_trace()
    for i in range(1, epoch_num + 1):
        print('---------------------------EPOCH: ' + str(i) + ' ---------------------------')
        print('---------------------------EPOCH: ' + str(i) + ' ---------------------------')
        print('---------------------------EPOCH: ' + str(i) + ' ---------------------------')
        print('---------------------------EPOCH: ' + str(i) + ' ---------------------------')
        print('---------------------------EPOCH: ' + str(i) + ' ---------------------------')
        model.train()
        epoch_ypred = np.zeros((1, 1))
        upper_index = 0
        batch_loss_list = []
        dl_input_num = xTr.shape[0]
        for index in range(0, dl_input_num, batch_size):
            if (index + batch_size) < dl_input_num:
                upper_index = index + batch_size
            else:
                upper_index = dl_input_num
            geo_datalist = read_batch(index, upper_index, xTr, yTr, drugTr,\
                num_feature, num_gene, num_drug, edge_index, dataset)
            dataset_loader, node_num, feature_dim = GeoGraphLoader.load_graph(geo_datalist, prog_args)
            # ACTIVATE LEARNING RATE SCHEDULE
            iteration_num += 1
            learning_rate = learning_rate_schedule(args, dl_input_num, iteration_num, e1, e2, e3, e4)
            # learning_rate = 0.001
            print('TRAINING MODEL...')
            model, batch_loss, batch_ypred = train_geogat_model(dataset_loader, model, device, args, learning_rate)
            print('BATCH LOSS: ', batch_loss)
            batch_loss_list.append(batch_loss)
            # PRESERVE PREDICTION OF BATCH TRAINING DATA
            batch_ypred = (Variable(batch_ypred).data).cpu().numpy()
            epoch_ypred = np.vstack((epoch_ypred, batch_ypred))
        epoch_loss = np.mean(batch_loss_list)
        print('TRAIN EPOCH ' + str(i) + ' MSE LOSS: ', epoch_loss)
        epoch_loss_list.append(epoch_loss)
        epoch_ypred = np.delete(epoch_ypred, 0, axis = 0)
        print(epoch_ypred)
        print('ITERATION NUMBER UNTIL NOW: ' + str(iteration_num))
        # PRESERVE PEARSON CORR FOR EVERY EPOCH
        score_lists = list(yTr)
        score_list = [item for elem in score_lists for item in elem]
        epoch_ypred_lists = list(epoch_ypred)
        epoch_ypred_list = [item for elem in epoch_ypred_lists for item in elem]
        train_dict = {'Score': score_list, 'Pred Score': epoch_ypred_list}
        tmp_training_input_df = pd.DataFrame(train_dict)
        # pdb.set_trace()
        epoch_pearson = tmp_training_input_df.corr(method='pearson')
        epoch_pearson_list.append(epoch_pearson['Pred Score'][0])
        tmp_training_input_df.to_csv(path + '/TrainingPred_' + str(i) + '.txt', index=False, header=True)
        print('EPOCH ' + str(i) + ' PEARSON CORRELATION: ', epoch_pearson)
        print('\n-------------EPOCH TRAINING PEARSON CORRELATION LIST: -------------')
        print(epoch_pearson_list)
        print('\n-------------EPOCH TRAINING MSE LOSS LIST: -------------')
        print(epoch_loss_list)
        epoch_pearson_array = np.array(epoch_pearson_list)
        epoch_loss_array = np.array(epoch_loss_list)
        np.save(path + '/pearson.npy', epoch_pearson_array)
        np.save(path + '/loss.npy', epoch_loss_array)

        # # # TEST MODEL ON TEST DATASET
        # fold_n = 1
        test_save_path = path
        test_pearson, test_loss, tmp_test_input_df = test_geogat(prog_args, fold_n, model, test_save_path, device, dataset)
        test_pearson_score = test_pearson['Pred Score'][0]
        test_pearson_list.append(test_pearson_score)
        test_loss_list.append(test_loss)
        tmp_test_input_df.to_csv(path + '/TestPred' + str(i) + '.txt', index = False, header = True)
        print('\n-------------EPOCH TEST PEARSON CORRELATION LIST: -------------')
        print(test_pearson_list)
        print('\n-------------EPOCH TEST MSE LOSS LIST: -------------')
        print(test_loss_list)
        # SAVE BEST TEST MODEL
        if test_pearson_score > max_test_corr:
            max_test_corr = test_pearson_score
            max_test_corr_id = i
            # torch.save(model.state_dict(), path + '/best_train_model'+ str(i) +'.pt')
            torch.save(model.state_dict(), path + '/best_train_model.pt')
        print('\n-------------BEST TEST PEARSON CORR MODEL ID INFO:' + str(max_test_corr_id) + '-------------')
        print('--- TRAIN ---')
        print('BEST MODEL TRAIN LOSS: ', epoch_loss_list[max_test_corr_id - 1])
        print('BEST MODEL TRAIN PEARSON CORR: ', epoch_pearson_list[max_test_corr_id - 1])
        print('--- TEST ---')
        print('BEST MODEL TEST LOSS: ', test_loss_list[max_test_corr_id - 1])
        print('BEST MODEL TEST PEARSON CORR: ', test_pearson_list[max_test_corr_id - 1])
        torch.save(model.state_dict(), path + '/best_train_model.pt')


def test_geogat_model(dataset_loader, model, device, args):
    batch_loss = 0
    for batch_idx, data in enumerate(dataset_loader):
        x = Variable(data.x, requires_grad=False).to(device)
        edge_index = Variable(data.edge_index, requires_grad=False).to(device)
        drug_index = Variable(data.drug_index, requires_grad=False).to(device)
        label = Variable(data.label, requires_grad=True).to(device)
        # THIS WILL USE METHOD [def forward()] TO MAKE PREDICTION
        ypred = model(x, edge_index, drug_index)
        loss = model.loss(ypred, label)
        batch_loss += loss.item()
    return model, batch_loss, ypred


def test_geogat(args, fold_n, model, test_save_path, device, dataset):
    print('-------------------------- TEST START --------------------------')
    print('-------------------------- TEST START --------------------------')
    print('-------------------------- TEST START --------------------------')
    print('-------------------------- TEST START --------------------------')
    print('-------------------------- TEST START --------------------------')
    # TEST MODEL ON TEST DATASET
    form_data_path = './' + dataset + '/form_data'
    xTe = np.load(form_data_path + '/xTe' + str(fold_n) + '.npy')
    yTe = np.load(form_data_path + '/yTe' + str(fold_n) + '.npy')
    drugTe =  np.load('./' + dataset + '/form_data/drugTe' + str(fold_n) + '.npy')
    edge_index = torch.from_numpy(np.load(form_data_path + '/edge_index.npy') ).long() 

    dl_input_num = xTe.shape[0]
    batch_size = args.batch_size
    # CLEAN RESULT PREVIOUS EPOCH_I_PRED FILES
    path = test_save_path
    # [num_feature, num_gene, num_drug]
    num_feature = 8
    dict_drug_num = pd.read_csv('./' + dataset + '/filtered_data/drug_num_dict.csv')
    num_drug = dict_drug_num.shape[0]
    final_annotation_gene_df = pd.read_csv('./' + dataset + '/filtered_data/kegg_gene_annotation.csv')
    num_gene = final_annotation_gene_df.shape[0]
    # RUN TEST MODEL
    model.eval()
    all_ypred = np.zeros((1, 1))
    upper_index = 0
    batch_loss_list = []
    for index in range(0, dl_input_num, batch_size):
        if (index + batch_size) < dl_input_num:
            upper_index = index + batch_size
        else:
            upper_index = dl_input_num
        geo_datalist = read_batch(index, upper_index, xTe, yTe, drugTe,\
                num_feature, num_gene, num_drug, edge_index, dataset)
        dataset_loader, node_num, feature_dim = GeoGraphLoader.load_graph(geo_datalist, prog_args)
        print('TEST MODEL...')
        # import pdb; pdb.set_trace()
        model, batch_loss, batch_ypred = test_geogat_model(dataset_loader, model, device, args)
        print('BATCH LOSS: ', batch_loss)
        batch_loss_list.append(batch_loss)
        # PRESERVE PREDICTION OF BATCH TEST DATA
        batch_ypred = (Variable(batch_ypred).data).cpu().numpy()
        all_ypred = np.vstack((all_ypred, batch_ypred))
    test_loss = np.mean(batch_loss_list)
    print('MSE LOSS: ', test_loss)
    # PRESERVE PEARSON CORR FOR EVERY EPOCH
    all_ypred = np.delete(all_ypred, 0, axis = 0)
    all_ypred_lists = list(all_ypred)
    all_ypred_list = [item for elem in all_ypred_lists for item in elem]
    score_lists = list(yTe)
    score_list = [item for elem in score_lists for item in elem]
    test_dict = {'Score': score_list, 'Pred Score': all_ypred_list}
    tmp_test_input_df = pd.DataFrame(test_dict)
    test_pearson = tmp_test_input_df.corr(method = 'pearson')
    print('PEARSON CORRELATION: ', test_pearson)
    print('FOLD - ', fold_n)
    return test_pearson, test_loss, tmp_test_input_df



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
    # dataset = 'datainfo-nci'
    dataset = 'datainfo-oneil'
 
    # ### TRAIN THE MODEL
    # TRAIN [FOLD-1]
    fold_n = 5
    # prog_args.model = 'load'
    # load_path = './' + dataset + '/result/epoch_60/best_train_model.pt'
    load_path = ''
    yTr = np.load('./' + dataset + '/form_data/yTr' + str(fold_n) + '.npy')
    # yTr = np.load('./' + dataset + '/form_data/y_split1.npy')
    dl_input_num = yTr.shape[0]
    epoch_iteration = int(dl_input_num / prog_args.batch_size)
    start_iter_num = 100 * epoch_iteration
    train_geogat(prog_args, fold_n, load_path, start_iter_num, device, dataset)

    # ### TEST THE MODEL
    # # TEST [FOLD-1]
    # fold_n = 1
    # model = build_geogat_model(prog_args, device, dataset)
    # test_load_path = './' + dataset + '/result/epoch_60/best_train_model.pt'
    # model.load_state_dict(torch.load(test_load_path, map_location=device))
    # test_save_path = './' + dataset + '/result/epoch_60'
    # test_geogat(prog_args, fold_n, model, test_save_path, device, dataset)