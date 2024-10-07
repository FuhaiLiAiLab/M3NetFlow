import os
import pdb
import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt
from torch.nn import init
from torch.nn.parameter import Parameter

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn.inits import glorot, zeros,kaiming_uniform

class WeBConv(MessagePassing):
    def __init__(self, in_channels, out_channels, node_num, num_edge, num_gene_edge, device):
        super(WeBConv, self).__init__(aggr='add')
        self.node_num = node_num
        self.num_edge = num_edge
        self.num_gene_edge = num_gene_edge
        self.num_drug_edge = num_edge - num_gene_edge

        self.up_proj = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.down_proj = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.bias_proj = torch.nn.Linear(in_channels, out_channels, bias=False)

        ##### [edge_weight] FOR ALL EDGES IN ONE [(gene+drug)] GRAPH #####
        ### [up_gene_edge_weight] [num_gene_edge / 21729] ###
        up_std_gene_edge = torch.nn.init.calculate_gain('relu')
        self.up_gene_edge_weight = torch.nn.Parameter((torch.randn(self.num_gene_edge) * up_std_gene_edge).to(device))
        ### [down_gene_edge_weight] [num_gene_edge / 21729] ###
        down_std_gene_edge = torch.nn.init.calculate_gain('relu')
        self.down_gene_edge_weight = torch.nn.Parameter((torch.randn(self.num_gene_edge) * down_std_gene_edge).to(device))


    def forward(self, x, edge_index):
        # [batch_size]
        batch_size = int(x.shape[0] / self.node_num)
        # TEST PARAMETERS
        print(torch.sum(self.up_gene_edge_weight))
        print(torch.sum(self.down_gene_edge_weight))

        ### [edge_index, x] ###
        up_edge_index = edge_index
        up_x = self.up_proj(x)
        down_edge_index = torch.flipud(edge_index)
        down_x = self.down_proj(x)
        bias_x = self.bias_proj(x)

        ### [edge_weight] ###
        # [up_edge_weight] = [up_gene_edge_weight] + [up_drug_edge_weight] [21729+116=21845]
        up_drug_edge_weight = torch.ones(self.num_drug_edge).to(device='cuda') # [/58*2=116]
        up_edge_weight = torch.cat((self.up_gene_edge_weight, up_drug_edge_weight), 0)
        # [down_edge_weight] = [down_gene_edge_weight] + [down_drug_edge_weight] [21729+116=21845]
        down_drug_edge_weight = torch.ones(self.num_drug_edge).to(device='cuda')
        down_edge_weight = torch.cat((self.down_gene_edge_weight, down_drug_edge_weight), 0) # [/58*2=116]
        # [batch_up/down_edge_weight] [N*21845]
        batch_up_edge_weight = up_edge_weight.repeat(1, batch_size)
        batch_down_edge_weight = down_edge_weight.repeat(1, batch_size)

        # Step 3: Compute normalization.
        # [row] FOR 1st LINE && [col] FOR 2nd LINE
        # [up]
        up_row, up_col = up_edge_index
        up_deg = degree(up_col, x.size(0), dtype=x.dtype)
        up_deg_inv_sqrt = up_deg.pow(-1)
        up_deg_inv_sqrt[up_deg_inv_sqrt == float('inf')] = 0
        up_norm = up_deg_inv_sqrt[up_col]
        # [down]
        down_row, down_col = down_edge_index
        down_deg = degree(down_col, x.size(0), dtype=x.dtype)
        down_deg_inv_sqrt = down_deg.pow(-1)
        down_deg_inv_sqrt[down_deg_inv_sqrt == float('inf')] = 0
        down_norm = down_deg_inv_sqrt[down_col]
        # Check [ torch.sum(up_norm[0:21729]==up_norm[21845:43574])==21729 ]

        # Step 4-5: Start propagating messages.
        x_up = self.propagate(up_edge_index, x=up_x, norm=up_norm, edge_weight=batch_up_edge_weight)
        x_down = self.propagate(down_edge_index, x=down_x, norm=down_norm, edge_weight=batch_down_edge_weight)
        x_bias = bias_x
        concat_x = torch.cat((x_up, x_down, x_bias), dim=-1)
        concat_x = F.normalize(concat_x, p=2, dim=-1)
        return concat_x, up_edge_weight, down_edge_weight

    def message(self, x_j, norm, edge_weight):
        # [x_j] has shape [E, out_channels]
        # Step 4: Normalize node features.
        weight_norm = torch.mul(norm, edge_weight)
        return weight_norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # [aggr_out] has shape [N, out_channels]
        return aggr_out


class SubGraphAttentionConv(MessagePassing):
    def __init__(self, in_channels, out_channels, head, negative_slope, aggr, device, dataset):
        super(SubGraphAttentionConv, self).__init__(node_dim=0)
        assert out_channels % head == 0
        self.k = out_channels // head

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_head = head
        self.negative_slope = negative_slope
        self.aggr = aggr
        self.device = device
        self.dataset = dataset

        self.weight_linear = nn.Linear(in_channels, out_channels, bias=False)
        self.att = torch.nn.Parameter(torch.Tensor(1, head, 2 * self.k))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_linear.weight.data)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, mask, batch_random_final_dl_input_df, sp_path):
        dataset = self.dataset
        ### ADD SELF LOOPS IN THE EDGE SPACE
        x = self.weight_linear(x).view(-1, self.num_head, self.k) # N * num_head * h
        # import pdb; pdb.set_trace()
        batch_size = batch_random_final_dl_input_df.shape[0]
        sub_khop_node_num = int(x.shape[0] / batch_size)
        return self.propagate(edge_index, x=x, mask=mask, batch_df=batch_random_final_dl_input_df, path=sp_path, sub_khop_node_num=sub_khop_node_num, dataset=dataset)

    def message(self, edge_index, x_i, x_j, mask, batch_df, path, sub_khop_node_num, dataset):
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1) # E * num_head * h
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])
        head_mask = torch.tile(mask, (1, self.num_head)).reshape(x_j.shape[0], self.num_head, 1)
        x_j.masked_fill_(head_mask==0, 0.)

        # CACULATE THE ATTENTION
        batch_size = batch_df.shape[0]
        batch_alpha = alpha.clone()
        mask = mask.unsqueeze(-1)
        batch_mask = mask.reshape(batch_size, -1)
        batch_alpha = batch_alpha.masked_fill_(mask==0, 0.)
        batch_alpha = torch.mean(batch_alpha, dim=1)
        batch_alpha = batch_alpha.reshape(batch_size, -1)
        batch_cell_line_list = list(batch_df['Cell Line Name'])
        batch_edgeindex = edge_index.clone()
        batch_edgeindex = batch_edgeindex.t().reshape(batch_size, -1, 2)

        # PREPARE [cell line] MAP
        cell_line_map_df = pd.read_csv('./' + dataset + '/filtered_data/cell_line_map_dict.csv')
        cell_line_map_dict = dict(zip(cell_line_map_df.Cell_Line_Name, cell_line_map_df.Cell_Line_Num))

        # import pdb; pdb.set_trace()

        for batch_idx in range(batch_size):
            input_row = batch_df.iloc[batch_idx]
            drugA = input_row[0]
            drugB = input_row[1]
            cell_line = input_row[2]
            cell_line_num = cell_line_map_dict[cell_line]
            cell_line_save_path = path + '/cell' + str(cell_line_num) + '.csv'
            if os.path.isfile(cell_line_save_path) == True:
                continue

            # import pdb; pdb.set_trace()
            
            hop_edge_num = int((torch.sum(batch_mask[batch_idx])).cpu().detach().numpy())
            hop1_list = ['hop1' for x in range(hop_edge_num)]
            hop2_list = ['hop2' for x in range(hop_edge_num)]
            hop3_list = ['hop3' for x in range(hop_edge_num)]
            hop_list = hop1_list + hop2_list + hop3_list


            from_array = batch_edgeindex[batch_idx, :, 0] - (batch_idx * sub_khop_node_num)
            to_array = batch_edgeindex[batch_idx, :, 1] - (batch_idx * sub_khop_node_num)

            from_list =  list(from_array.cpu().detach().numpy())
            to_list =  list(to_array.cpu().detach().numpy())
            mask_list = list(batch_mask[batch_idx].cpu().detach().numpy())
            attention_list = list(batch_alpha[batch_idx].cpu().detach().numpy())
            cell_line_att_df = pd.DataFrame({'From': from_list,
                                             'To': to_list,
                                             'Mask': mask_list,
                                             'Attention': attention_list,
                                             'Hop': hop_list})
            # import pdb; pdb.set_trace()
            cell_line_att_df.to_csv(cell_line_save_path, index=False, header=True)

        return x_j * alpha.view(-1, self.num_head, 1) # E * num_head * h

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.out_channels)
        aggr_out = aggr_out + self.bias
        return aggr_out


class TraverseSubGNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, head, max_layer, device, dataset):
        super(TraverseSubGNN, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.max_layer = max_layer
        self.device = device
        self.dataset = dataset

        self.subgat_khop = self.build_traverse_subgat_layer(
                            input_dim=input_dim, embedding_dim=embedding_dim, head=head, max_layer=max_layer, device=device)
        self.act2 = nn.LeakyReLU(negative_slope=0.1)
        self.norm = nn.BatchNorm1d(embedding_dim)

    def build_traverse_subgat_layer(self, input_dim, embedding_dim, head, max_layer, device):
        subgat_khop = SubGraphAttentionConv(in_channels=input_dim, out_channels=embedding_dim, head=3, negative_slope=0.2, aggr="add", device=device, dataset=self.dataset)
        return subgat_khop
    
    def forward(self, subx, subadj_edgeindex, sub_mask_edgeindex, batch_size, subgraph_size, batch_random_final_dl_input_df, sp_path):
        khop_subx = self.subgat_khop(subx, subadj_edgeindex, sub_mask_edgeindex, batch_random_final_dl_input_df, sp_path)
        khop_subx = self.norm(self.act2(khop_subx))
        khop_subx = khop_subx.reshape(batch_size, self.max_layer, subgraph_size, -1)
        khop_subx = torch.mean(khop_subx, dim=1)
        return khop_subx


class GlobalWeBGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim,
                            node_num, num_edge, num_gene_edge, device):
        super(GlobalWeBGNN, self).__init__()
        self.node_num = node_num
        self.num_edge = num_edge
        self.num_gene_edge = num_gene_edge

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.device = device
        self.webconv_first, self.webconv_block, self.webconv_last = self.build_webconv_layer(
                    input_dim, hidden_dim, embedding_dim, node_num, num_edge, num_gene_edge)
        
        self.act = nn.ReLU()
        self.act2 = nn.LeakyReLU(negative_slope=0.1)


    def build_webconv_layer(self, input_dim, hidden_dim, embedding_dim, node_num, num_edge, num_gene_edge):
        # webconv_first [input_dim, hidden_dim]
        webconv_first = WeBConv(in_channels=input_dim, out_channels=hidden_dim,
                node_num=node_num, num_edge=num_edge, num_gene_edge=num_gene_edge, device=self.device)
        # webconv_block [hidden_dim*3, hidden_dim]
        webconv_block = WeBConv(in_channels=int(hidden_dim*3), out_channels=hidden_dim,
                node_num=node_num, num_edge=num_edge, num_gene_edge=num_gene_edge, device=self.device)
        # webconv_last [hidden_dim*3, embedding_dim]
        webconv_last = WeBConv(in_channels=int(hidden_dim*3), out_channels=embedding_dim,
                node_num=node_num, num_edge=num_edge, num_gene_edge=num_gene_edge, device=self.device)
        return webconv_first, webconv_block, webconv_last

    def forward(self, x, edge_index):
        # webconv_first
        web_x, first_up_edge_weight, first_down_edge_weight = self.webconv_first(x, edge_index)
        web_x = self.act2(web_x)
        # webconv_block
        web_x, block_up_edge_weight, block_down_edge_weight = self.webconv_block(web_x, edge_index)
        web_x = self.act2(web_x)
        # webconv_last
        web_x, last_up_edge_weight, last_down_edge_weight = self.webconv_last(web_x, edge_index)
        web_x = self.act2(web_x)
        # [mean_up_edge_weight / mean_down_edge_weight]
        mean_up_edge_weight = (1/3) * (first_up_edge_weight + block_up_edge_weight + last_up_edge_weight)
        mean_down_edge_weight = (1/3) * (first_down_edge_weight + block_down_edge_weight + last_down_edge_weight)
        return web_x, mean_up_edge_weight, mean_down_edge_weight


class M3NetFlowDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, decoder_dim,
                                num_gene, node_num, num_edge, num_gene_edge, device, dataset):
        super(M3NetFlowDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.device = device
        self.dataset = dataset

        self.num_gene = num_gene
        self.node_num = node_num
        self.num_edge = num_edge
        self.num_gene_edge = num_gene_edge

        self.max_layer = 3
        self.traverse_subgnn = TraverseSubGNN(input_dim=input_dim, embedding_dim=input_dim*3, head=3, max_layer=self.max_layer, device=device, dataset=dataset)
        self.linear_traverse_x = nn.Linear(input_dim*3, 1)
        self.linear_x = nn.Linear(input_dim, input_dim)
        
        self.x_norm = nn.BatchNorm1d(input_dim)


        ##### GLOBAL PROPAGATION LAYERS
        self.global_gnn = GlobalWeBGNN(input_dim=input_dim+1, hidden_dim=hidden_dim, embedding_dim=embedding_dim,
                                node_num=node_num, num_edge=num_edge, num_gene_edge=num_gene_edge, device=device)

        self.parameter1 = torch.nn.Parameter(torch.randn(int(embedding_dim*3), decoder_dim).to(device='cuda'))
        self.parameter2 = torch.nn.Parameter(torch.randn(decoder_dim, decoder_dim).to(device='cuda'))

    
    def forward(self, x, edge_index, drug_index, adj, batch_random_final_dl_input_df, analysis_save_path):
        dataset = self.dataset
        ### BUILD UP ASSIGNMENT MATRIX
        x_norm = self.x_norm(x)
        x = x.reshape(-1, self.node_num, self.input_dim)
        x_norm = x_norm.reshape(-1, self.node_num, self.input_dim)
        gene_x = x[:, :self.num_gene, :]
        
        ### TRAVERSE SUBGRAPH
        # FORM NOTATION TO [signaling pathways]
        form_data_path = './' + dataset + '/form_data'
        kegg_path_gene_interaction_df = pd.read_csv('./' + dataset + '/filtered_data/kegg_path_gene_interaction.csv')
        kegg_sp_list = list(set(kegg_path_gene_interaction_df['path']))
        kegg_sp_list.sort()
        kegg_sp_notation_list = ['sp' + str(x) for x in range(1, len(kegg_sp_list)+1)]
        kegg_num_notation_list = [x for x in range(len(kegg_sp_list))]
        
        # FOR LOOP TO START TRAVERSE
        batch_size = x.shape[0] # batch_size
        # INITIALIZE [traverse_x]
        traverse_sp_x = torch.zeros(x.shape[0], len(kegg_sp_list), self.node_num, self.input_dim*3).to(device='cuda')
        traverse_spnum_sum = torch.zeros([self.node_num, 1]).to(device='cuda')

        for sp_num_notation in kegg_num_notation_list:
            # ASSIGN CERTAIN SUBGRAPH NODE
            sp_notation = kegg_sp_notation_list[sp_num_notation]
            subassign_index = np.load(form_data_path + '/' + sp_notation + '_gene_idx.npy')
            subgraph_size = subassign_index.shape[0]
            # SP PATH CREATION
            sp_path = analysis_save_path + '/' + sp_notation
            while os.path.exists(sp_path) == False:
                os.mkdir(sp_path)
            # EXPAND NODE TO MULTIPLE HOPs
            subx = gene_x[:, subassign_index, :]
            batch_subx = torch.tile(subx, (1, self.max_layer, 1))
            batch_subx = batch_subx.reshape(-1, subx.shape[2])
            # GET [sp_adj_edgeindex, sp_mask_edgeindex]
            sp_notation = kegg_sp_notation_list[sp_num_notation]
            sp_adj_edgeindex = np.load(form_data_path + '/' + sp_notation + '_khop_subadj_edgeindex.npy')
            sp_adj_edgeindex = torch.from_numpy(sp_adj_edgeindex).to(device='cuda')
            sp_mask_edgeindex = np.load(form_data_path + '/' + sp_notation + '_khop_mask_edgeindex.npy')
            sp_mask_edgeindex = torch.from_numpy(sp_mask_edgeindex).to(device='cuda')
            # EXPAND [adj_edgeindex, mask_edgeindex] TO [batch_size]
            tmp_sp_adj_edgeindex = sp_adj_edgeindex.clone()
            batch_sp_adj_edgeindex = sp_adj_edgeindex
            batch_sp_mask_edgeindex = sp_mask_edgeindex
            for batch_idx in range(2, batch_size + 1):
                tmp_sp_adj_edgeindex += (subgraph_size) * (self.max_layer)
                batch_sp_adj_edgeindex = torch.cat([batch_sp_adj_edgeindex, tmp_sp_adj_edgeindex], dim=1)
                batch_sp_mask_edgeindex = torch.cat([batch_sp_mask_edgeindex, sp_mask_edgeindex])
            # RUN [traverse_subgnn]
            khop_subx = self.traverse_subgnn(batch_subx, batch_sp_adj_edgeindex, batch_sp_mask_edgeindex, 
                                    batch_size, subgraph_size, batch_random_final_dl_input_df, sp_path)
            traverse_sp_x[:, sp_num_notation, subassign_index, :] = khop_subx
            traverse_sp_tmp_num = torch.zeros([self.node_num, 1]).to(device='cuda')
            traverse_sp_tmp_num[subassign_index, :] = 1
            traverse_spnum_sum += traverse_sp_tmp_num

        # import pdb; pdb.set_trace()
        
        traverse_sum_x = torch.sum(traverse_sp_x, axis=1)
        traverse_spnum_sum = torch.tile(traverse_spnum_sum, (batch_size, 1, 1))
        traverse_x = torch.div(traverse_sum_x, traverse_spnum_sum)
        traverse_x = torch.nan_to_num(traverse_x, nan=0)
        
        # import pdb; pdb.set_trace()
        transformed_traverse_x = self.linear_traverse_x(traverse_x)
        # transformed_x = self.linear_x(x)

        # # USE RES-NET IDEA
        # transformed_traverse_x = transformed_traverse_x.reshape(-1, self.input_dim)
        # norm_transformed_traverse_x = self.x_norm(transformed_traverse_x)
        # norm_transformed_traverse_x = norm_transformed_traverse_x.reshape(-1, self.node_num, self.input_dim)
        # global_x = x + norm_transformed_traverse_x

        global_x = torch.cat((x, transformed_traverse_x), dim=-1)
        # global_x = torch.cat((transformed_x, transformed_traverse_x), dim=-1)
        # global_x = torch.cat((x_norm, traverse_x), dim=-1)
        global_x = global_x.reshape(-1, global_x.shape[2])
        global_x, global_mean_up_edge_weight, global_mean_down_edge_weight = self.global_gnn(global_x, edge_index)
        final_x = global_x
        # import pdb; pdb.set_trace()

        drug_index = torch.reshape(drug_index, (-1, 2))

        # EMBEDDING DECODER TO [ypred]
        batch_size, drug_num = drug_index.shape
        ypred = torch.zeros(batch_size, 1).to(device='cuda')
        for i in range(batch_size):
            drug_a_idx = int(drug_index[i, 0]) - 1
            drug_b_idx = int(drug_index[i, 1]) - 1
            drug_a_embedding = final_x[drug_a_idx]
            drug_b_embedding = final_x[drug_b_idx]
            product1 = torch.matmul(drug_a_embedding, self.parameter1)
            product2 = torch.matmul(product1, self.parameter2)
            product3 = torch.matmul(product2, torch.transpose(self.parameter1, 0, 1))
            output = torch.matmul(product3, drug_b_embedding.reshape(-1, 1))
            ypred[i] = output
        print(self.parameter1)
        print(torch.sum(self.parameter1))
        # print(self.parameter2)
        return ypred

    def loss(self, pred, label):
        # import pdb; pdb.set_trace()
        pred = pred.to(device='cuda')
        label = label.to(device='cuda')
        loss = F.mse_loss(pred.squeeze(), label)
        # print(pred)
        # print(label)
        return loss