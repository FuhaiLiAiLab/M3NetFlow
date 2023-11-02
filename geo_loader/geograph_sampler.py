import torch
import numpy as np
import networkx as nx
import torch.utils.data

from torch_geometric.data import Batch
from torch_geometric.data import DataListLoader

class GeoGraphLoader():
    def __init__(self):
        pass

    def load_graph(geo_datalist, args):
        dataset_sampler = GeoGraphSampler(geo_datalist)
        dataset_loader = torch.utils.data.DataLoader(
                                        dataset=dataset_sampler, 
                                        batch_size=args.batch_size, 
                                        shuffle=False,
                                        num_workers=args.num_workers,
                                        collate_fn=collate_fn)
        return dataset_loader, dataset_sampler.node_num, dataset_sampler.feature_dim

# CUSTOM GRAPH DATASET WITH [init, len, getitem]
class GeoGraphSampler(torch.utils.data.Dataset):
    # INITIATE WITH [node_num, feature_dim, length, geo_batch_datalist] FROM [geo_datalist]
    def __init__(self, geo_datalist):
        self.node_num = geo_datalist[0].x.shape[0]
        self.feature_dim = geo_datalist[0].x.shape[1]
        self.length = len(geo_datalist)
        self.geo_batch_datalist = Batch.from_data_list(geo_datalist)

    def __len__(self):
        return self.length

    # INPUT POINT: A GRAPH (ALL OF NODES, AND ITS NODES FEATURES)
    # INPUT LABEL: A GRAPH LABEL
    def __getitem__(self, idx):
        return self.geo_batch_datalist[idx]

def collate_fn(examples):
    data_list = Batch.from_data_list(examples)
    return (data_list)