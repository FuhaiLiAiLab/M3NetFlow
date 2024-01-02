import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree, softmax

from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch import Tensor
from torch.nn import ModuleList, Sequential
from torch.utils.data import DataLoader

from torch_geometric.nn.aggr import DegreeScalerAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import degree


class PNAConv(MessagePassing):
    r"""The Principal Neighbourhood Aggregation graph convolution operator
    from the `"Principal Neighbourhood Aggregation for Graph Nets"
    <https://arxiv.org/abs/2004.05718>`_ paper.

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left(
        \mathbf{x}_i, \underset{j \in \mathcal{N}(i)}{\bigoplus}
        h_{\mathbf{\Theta}} \left( \mathbf{x}_i, \mathbf{x}_j \right)
        \right)

    with

    .. math::
        \bigoplus = \underbrace{\begin{bmatrix}
            1 \\
            S(\mathbf{D}, \alpha=1) \\
            S(\mathbf{D}, \alpha=-1)
        \end{bmatrix} }_{\text{scalers}}
        \otimes \underbrace{\begin{bmatrix}
            \mu \\
            \sigma \\
            \max \\
            \min
        \end{bmatrix}}_{\text{aggregators}},

    where :math:`\gamma_{\mathbf{\Theta}}` and :math:`h_{\mathbf{\Theta}}`
    denote MLPs.

    .. note::

        For an example of using :obj:`PNAConv`, see `examples/pna.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/
        examples/pna.py>`_.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        aggregators (List[str]): Set of aggregation function identifiers,
            namely :obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"var"` and :obj:`"std"`.
        scalers (List[str]): Set of scaling function identifiers, namely
            :obj:`"identity"`, :obj:`"amplification"`,
            :obj:`"attenuation"`, :obj:`"linear"` and
            :obj:`"inverse_linear"`.
        deg (torch.Tensor): Histogram of in-degrees of nodes in the training
            set, used by scalers to normalize.
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default :obj:`None`)
        towers (int, optional): Number of towers (default: :obj:`1`).
        pre_layers (int, optional): Number of transformation layers before
            aggregation (default: :obj:`1`).
        post_layers (int, optional): Number of transformation layers after
            aggregation (default: :obj:`1`).
        divide_input (bool, optional): Whether the input features should
            be split between towers or not (default: :obj:`False`).
        act (str or callable, optional): Pre- and post-layer activation
            function to use. (default: :obj:`"relu"`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        train_norm (bool, optional): Whether normalization parameters
            are trainable. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggregators: List[str],
        scalers: List[str],
        deg: Tensor,
        edge_dim: Optional[int] = None,
        towers: int = 1,
        pre_layers: int = 1,
        post_layers: int = 1,
        divide_input: bool = False,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        train_norm: bool = False,
        **kwargs,
    ):

        aggr = DegreeScalerAggregation(aggregators, scalers, deg, train_norm)
        super().__init__(aggr=aggr, node_dim=0, **kwargs)

        if divide_input:
            assert in_channels % towers == 0
        assert out_channels % towers == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.towers = towers
        self.divide_input = divide_input

        self.F_in = in_channels // towers if divide_input else in_channels
        self.F_out = self.out_channels // towers

        if self.edge_dim is not None:
            self.edge_encoder = Linear(edge_dim, self.F_in)

        self.pre_nns = ModuleList()
        self.post_nns = ModuleList()
        for _ in range(towers):
            modules = [Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [activation_resolver(act, **(act_kwargs or {}))]
                modules += [Linear(self.F_in, self.F_in)]
            self.pre_nns.append(Sequential(*modules))

            in_channels = (len(aggregators) * len(scalers) + 1) * self.F_in
            modules = [Linear(in_channels, self.F_out)]
            for _ in range(post_layers - 1):
                modules += [activation_resolver(act, **(act_kwargs or {}))]
                modules += [Linear(self.F_out, self.F_out)]
            self.post_nns.append(Sequential(*modules))

        self.lin = Linear(out_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.edge_dim is not None:
            self.edge_encoder.reset_parameters()
        for nn in self.pre_nns:
            reset(nn)
        for nn in self.post_nns:
            reset(nn)
        self.lin.reset_parameters()


    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:

        if self.divide_input:
            x = x.view(-1, self.towers, self.F_in)
        else:
            x = x.view(-1, 1, self.F_in).repeat(1, self.towers, 1)

        # propagate_type: (x: Tensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

        out = torch.cat([x, out], dim=-1)
        outs = [nn(out[:, i]) for i, nn in enumerate(self.post_nns)]
        out = torch.cat(outs, dim=1)

        return self.lin(out)


    def message(self, x_i: Tensor, x_j: Tensor,
                edge_attr: OptTensor) -> Tensor:

        h: Tensor = x_i  # Dummy.
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
            edge_attr = edge_attr.view(-1, 1, self.F_in)
            edge_attr = edge_attr.repeat(1, self.towers, 1)
            h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)

        hs = [nn(h[:, i]) for i, nn in enumerate(self.pre_nns)]
        return torch.stack(hs, dim=1)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, towers={self.towers}, '
                f'edge_dim={self.edge_dim})')

class PNADecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, decoder_dim, deg_histogram, device):
        super(PNADecoder, self).__init__()
        self.deg_histogram = deg_histogram
        self.embedding_dim = embedding_dim
        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layer(
                    input_dim, hidden_dim, embedding_dim, deg_histogram)
        
        self.act = nn.ReLU()
        self.act2 = nn.LeakyReLU(negative_slope=0.1)
        
        self.parameter1 = torch.nn.Parameter(torch.randn(embedding_dim, decoder_dim).to(device='cuda'))
        self.parameter2 = torch.nn.Parameter(torch.randn(decoder_dim, decoder_dim).to(device='cuda'))

        self.x_norm_first = nn.BatchNorm1d(hidden_dim)
        self.x_norm_block = nn.BatchNorm1d(hidden_dim)
        self.x_norm_last = nn.BatchNorm1d(embedding_dim)


    def build_conv_layer(self, input_dim, hidden_dim, embedding_dim, deg_histogram):
        conv_first = PNAConv(in_channels=input_dim, out_channels=hidden_dim, aggregators=['mean'], scalers=['identity'], deg=torch.ones(1, dtype=torch.long))
        conv_block = PNAConv(in_channels=hidden_dim, out_channels=hidden_dim, aggregators=['mean'], scalers=['identity'], deg=torch.ones(1, dtype=torch.long))
        conv_last = PNAConv(in_channels=hidden_dim, out_channels=embedding_dim, aggregators=['mean'], scalers=['identity'], deg=torch.ones(1, dtype=torch.long))
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