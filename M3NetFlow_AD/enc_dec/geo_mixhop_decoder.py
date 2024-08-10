import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional
from torch import Tensor
from torch.nn import Parameter

from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree, softmax

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import spmm
from torch_geometric.nn import aggr


class MixHopConv(MessagePassing):
    r"""The Mix-Hop graph convolutional operator from the
    `"MixHop: Higher-Order Graph Convolutional Architecturesvia Sparsified
    Neighborhood Mixing" <https://arxiv.org/abs/1905.00067>`_ paper.

    .. math::
        \mathbf{X}^{\prime}={\Bigg\Vert}_{p\in P}
        {\left( \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \right)}^p \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        powers (List[int], optional): The powers of the adjacency matrix to
            use. (default: :obj:`[0, 1, 2]`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:**
          node features :math:`(|\mathcal{V}|, |P| \cdot F_{out})`
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        powers: Optional[List[int]] = None,
        add_self_loops: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        if powers is None:
            powers = [0, 1, 2]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.powers = powers
        self.add_self_loops = add_self_loops

        self.lins = torch.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False)
            if p in powers else torch.nn.Identity()
            for p in range(max(powers) + 1)
        ])

        if bias:
            self.bias = Parameter(torch.empty(len(powers) * out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            if hasattr(lin, 'reset_parameters'):
                lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, self.flow, x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, self.flow, x.dtype)

        outs = [self.lins[0](x)]

        for lin in self.lins[1:]:
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                               size=None)

            outs.append(lin.forward(x))

        out = torch.cat([outs[p] for p in self.powers], dim=-1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, powers={self.powers})')


class MixhopDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_head, device, num_class, node_num):
        super(MixhopDecoder, self).__init__()
        self.num_head = num_head
        self.embedding_dim = embedding_dim
        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layer(
                    input_dim, hidden_dim, embedding_dim)
        
        self.num_class = num_class
        self.node_num = node_num
        
        self.act = nn.ReLU()
        self.act2 = nn.LeakyReLU(negative_slope=0.1)

        self.x_norm_first = nn.BatchNorm1d(hidden_dim*3)
        self.x_norm_block = nn.BatchNorm1d(hidden_dim*3)
        self.x_norm_last = nn.BatchNorm1d(embedding_dim*3)

        # Simple aggregations
        self.mean_aggr = aggr.MeanAggregation()
        self.max_aggr = aggr.MaxAggregation()
        # Learnable aggregations
        self.softmax_aggr = aggr.SoftmaxAggregation(learn=True)
        self.powermean_aggr = aggr.PowerMeanAggregation(learn=True)

        self.graph_prediction = torch.nn.Linear(embedding_dim*3, num_class)


    def build_conv_layer(self, input_dim, hidden_dim, embedding_dim):
        conv_first = MixHopConv(in_channels=input_dim, out_channels=hidden_dim)
        conv_block = MixHopConv(in_channels=hidden_dim*3, out_channels=hidden_dim)
        conv_last = MixHopConv(in_channels=hidden_dim*3, out_channels=embedding_dim)
        return conv_first, conv_block, conv_last

    def forward(self, x, edge_index):
        x = self.conv_first(x, edge_index)
        # import pdb; pdb.set_trace()
        x = self.x_norm_first(x)
        x = self.act2(x)

        x = self.conv_block(x, edge_index)
        x = self.x_norm_block(x)
        x = self.act2(x)

        x = self.conv_last(x, edge_index)
        x = self.x_norm_last(x)
        x = self.act2(x)
        
        # Embedding decoder to [ypred]
        x = x.view(-1, self.node_num, self.embedding_dim*3)
        x = self.powermean_aggr(x).view(-1, self.embedding_dim*3)
        output = self.graph_prediction(x)
        _, ypred = torch.max(output, dim=1)
        return output, ypred


    def loss(self, output, label):
        num_class = self.num_class
        # Use weight vector to balance the loss
        weight_vector = torch.zeros([num_class]).to(device='cuda')
        label = label.long()
        for i in range(num_class):
            n_samplei = torch.sum(label == i)
            if n_samplei == 0:
                weight_vector[i] = 0
            else:
                weight_vector[i] = len(label) / (n_samplei)
        # Calculate the loss
        output = torch.log_softmax(output, dim=-1)
        loss = F.nll_loss(output, label, weight_vector)
        return loss