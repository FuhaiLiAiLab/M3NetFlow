import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree, softmax


# GAT torch_geometric implementation
#Adapted from https://github.com/snap-stanford/pretrain-gnns
class GATConv(MessagePassing):
    def __init__(self, input_dim, embed_dim, num_head=1, negative_slope=0.2, aggr="add", num_edge_type=0):
        super(GATConv, self).__init__(node_dim=0)
        assert embed_dim % num_head == 0
        self.k = embed_dim // num_head
        self.aggr = aggr

        self.embed_dim = embed_dim
        self.num_head = num_head
        self.negative_slope = negative_slope

        self.weight_linear = nn.Linear(input_dim, embed_dim,bias=False)
        self.att = torch.nn.Parameter(torch.Tensor(1, num_head, 2 * num_head * self.k))
        self.bias = torch.nn.Parameter(torch.Tensor(embed_dim))

        if num_edge_type>0:
            self.edge_embedding = torch.nn.Embedding(num_edge_type, embed_dim)
            nn.init.xavier_uniform_(self.edge_embedding.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_linear.weight.data)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr=None):
        # import pdb; pdb.set_trace()
        #add self loops in the edge space
        edge_index,_ = add_self_loops(edge_index, num_nodes = x.size(0))
        x = self.weight_linear(x).view(-1, self.num_head, self.k) # N * num_head * k

        if edge_attr is not None:
            #add features corresponding to self-loop edges, set as zeros.
            self_loop_attr = torch.zeros(x.size(0),dtype=torch.long)
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

            edge_embeddings = self.edge_embedding(edge_attr)
            return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)
        else:
            return self.propagate(edge_index, x=x, edge_attr=None)

    def message(self, edge_index, x_i, x_j, edge_attr):
        if edge_attr is not None:
            edge_attr = edge_attr.view(-1, self.num_head, self.k)
            x_j += edge_attr
            
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1) # E * num_head
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])
        return x_j * alpha.view(-1, self.num_head, 1) #E * num_head * k

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1,self.embed_dim)

        aggr_out = aggr_out + self.bias

        return F.relu(aggr_out)


class GATDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, decoder_dim, num_head, device):
        super(GATDecoder, self).__init__()
        self.num_head = num_head
        self.embedding_dim = embedding_dim
        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layer(
                    input_dim, hidden_dim, embedding_dim)
        
        self.act = nn.ReLU()
        self.act2 = nn.LeakyReLU(negative_slope=0.1)
        
        self.parameter1 = torch.nn.Parameter(torch.randn(embedding_dim, decoder_dim).to(device='cuda'))
        self.parameter2 = torch.nn.Parameter(torch.randn(decoder_dim, decoder_dim).to(device='cuda'))

        self.x_norm_first = nn.BatchNorm1d(hidden_dim)
        self.x_norm_block = nn.BatchNorm1d(hidden_dim)
        self.x_norm_last = nn.BatchNorm1d(embedding_dim)


    def build_conv_layer(self, input_dim, hidden_dim, embedding_dim):
        conv_first = GATConv(input_dim=input_dim, embed_dim=hidden_dim, num_head=self.num_head)
        conv_block = GATConv(input_dim=hidden_dim, embed_dim=hidden_dim, num_head=self.num_head)
        conv_last = GATConv(input_dim=hidden_dim, embed_dim=embedding_dim, num_head=self.num_head)
        return conv_first, conv_block, conv_last

    def forward(self, x, edge_index, drug_index):
        x = self.conv_first(x, edge_index)
        x = self.x_norm_first(x)
        x = self.act2(x)

        x = self.conv_block(x, edge_index)
        x = self.x_norm_block(x)
        x = self.act2(x)

        x = self.conv_last(x, edge_index)
        x = self.x_norm_last(x)
        x = self.act2(x)
        
        # import pdb; pdb.set_trace()
        # x = torch.reshape(x, (-1, self.node_num, self.embedding_dim))
        drug_index = torch.reshape(drug_index, (-1, 2))

        # EMBEDDING DECODER TO [ypred]
        batch_size, drug_num = drug_index.shape
        ypred = torch.zeros(batch_size, 1).to(device='cuda')
        for i in range(batch_size):
            drug_a_idx = int(drug_index[i, 0]) - 1
            drug_b_idx = int(drug_index[i, 1]) - 1
            drug_a_embedding = x[drug_a_idx]
            drug_b_embedding = x[drug_b_idx]
            product1 = torch.matmul(drug_a_embedding, self.parameter1)
            product2 = torch.matmul(product1, self.parameter2)
            product3 = torch.matmul(product2, torch.transpose(self.parameter1, 0, 1))
            output = torch.matmul(product3, drug_b_embedding.reshape(-1, 1))
            ypred[i] = output
        print(torch.sum(self.parameter1))
        return ypred

    def loss(self, pred, label):
        pred = pred.to(device='cuda')
        label = label.to(device='cuda')
        loss = F.mse_loss(pred.squeeze(), label)
        # print(pred)
        # print(label)
        return loss