import torch
import numpy as np
import pandas as pd
import networkx as nx

from numpy import inf
from torch_geometric.data import Data

class ReadGeoGraph():
    def __init__(self):
        pass

    def read_feature(self, num_graph, num_feature, num_gene, num_drug, xBatch):
        # FORM [graph_feature_list]
        num_node = num_gene + num_drug
        xBatch = xBatch.reshape(num_graph, num_node, num_feature)
        graph_feature_list = []
        for i in range(num_graph):
            graph_feature_list.append(xBatch[i, :, :])
        return graph_feature_list

    def read_label(self, yBatch):
        yBatch_list = [label[0] for label in list(yBatch)]
        graph_label_list = yBatch_list
        return graph_label_list

    def read_drug(self, num_graph, drugBatch):
        graph_drug_list = []
        for i in range(num_graph):
            graph_drug_list.append(drugBatch[i, :])
        return graph_drug_list

    def form_geo_datalist(self, num_graph, graph_feature_list, graph_label_list, graph_drug_list, edge_index):
        geo_datalist = []
        for i in range(num_graph):
            graph_feature = graph_feature_list[i]
            graph_label = graph_label_list[i]
            graph_drug = graph_drug_list[i]
            # CONVERT [numpy] TO [torch]
            graph_feature = torch.from_numpy(graph_feature).float()
            graph_label = torch.from_numpy(np.array([graph_label])).float()
            graph_drug = torch.from_numpy(graph_drug).int()
            geo_data = Data(x=graph_feature, edge_index=edge_index, label=graph_label, drug_index=graph_drug)
            geo_datalist.append(geo_data)
        return geo_datalist


def read_batch(index, upper_index, x_input, y_input, drug_input,\
            num_feature, num_gene, num_drug, edge_index, dataset):
    # FORMING BATCH FILES
    form_data_path = './' + dataset + '/form_data'
    print('--------------' + str(index) + ' to ' + str(upper_index) + '--------------')
    xBatch = x_input[index : upper_index, :]
    yBatch = y_input[index : upper_index, :]
    drugBatch = drug_input[index : upper_index, :]
    print(xBatch.shape)
    print(yBatch.shape)
    print(drugBatch.shape)
    # PREPARE LOADING LISTS OF [features, labels, drugs, edge_index]
    print('READING BATCH GRAPHS TO LISTS ...')
    num_graph = upper_index - index
    # print('READING BATCH FEATURES ...')
    graph_feature_list =  ReadGeoGraph().read_feature(num_graph, num_feature, num_gene, num_drug, xBatch)
    # print('READING BATCH LABELS ...')
    graph_label_list = ReadGeoGraph().read_label(yBatch)
    # print('READING BATCH DRUGS ...')
    graph_drug_list = ReadGeoGraph().read_drug(num_graph, drugBatch)
    # print('FORMING GEOMETRIC GRAPH DATALIST ...')
    geo_datalist = ReadGeoGraph().form_geo_datalist(num_graph, \
        graph_feature_list, graph_label_list, graph_drug_list, edge_index)
    return geo_datalist



if __name__ == "__main__":
    # DATASET SELECTION
    dataset = 'datainfo-nci'
    # dataset = 'datainfo-oneil'

    # READ THESE FEATURE LABEL FILES
    x_split1 = np.load('./' + dataset + '/form_data/x_split1.npy')
    y_split1 = np.load('./' + dataset + '/form_data/y_split1.npy')
    drug_split1 = np.load('./' + dataset + '/form_data/drug_split1.npy')
    # READ [edge_index] FILES
    edge_index = np.load('./' + dataset + '/form_data/edge_index.npy')

    # [num_feature, num_gene, num_drug]
    num_feature = 8
    dict_drug_num = pd.read_csv('./' + dataset + '/filtered_data/dict_drug_num.csv')
    num_drug = dict_drug_num.shape[0]
    final_annotation_gene_df = pd.read_csv('./' + dataset + '/filtered_data/final_annotation_gene.csv')
    num_gene = final_annotation_gene_df.shape[0]
    # READ INTO LISTS
    geo_datalist = read_batch(0, 64, x_split1, y_split1, drug_split1,\
                num_feature, num_gene, num_drug, edge_index, dataset)
    # import pdb; pdb.set_trace()