library(shiny)
library(dplyr)
library(igraph)
library(networkD3)
library(kdensity)
library(ggplot2)
library(Jmisc)

net_edge = read.csv('./analysis-nci/cancer_cell_sp_edge.csv')
filtered_net_edge = filter(net_edge, Edge_type == 'cell-sp')
validated_net_edge = read.csv('./analysis-nci/validated_cancer_cell_sp_edge.csv')
filtered_validated_net_edge = filter(validated_net_edge, Edge_type == 'validated-cell-sp')

net_node = read.csv('./analysis-nci/cancer_cell_path_name_dict.csv')
filtered_net_node = filter(net_node, Type_name == 'Cell' | Type_name == 'SignalingPath')

net = graph_from_data_frame(d=filtered_net_edge, vertices=filtered_net_node, directed=F)

net_type = rep(TRUE, vcount(net))
net_type[V(net)$Type_name=='SignalingPath'] = FALSE
V(net)$type = net_type

### 4. NETWORK PARAMETERS SETTINGS
# vertex frame color
vertex_fcol = rep('black', vcount(net))
# vertex color
vertex_col = rep('lightblue', vcount(net))
vertex_col[V(net)$Type_name=='SignalingPath'] = 'gold'
# vertex size
vertex_size = rep(9.5, vcount(net))
vertex_size[V(net)$Type_name=='SignalingPath'] = 9.5
# vertex cex
vertex_cex = rep(0.8, vcount(net))
vertex_cex[V(net)$Type_name=='Cell'] = 0.7
# edge width
edge_width = rep(2.5, ecount(net))
edge_width[E(net)$Validation=='non-validated'] = 1
# edge color
edge_color = rep('gray', ecount(net))
edge_color[E(net)$Validation=='non-validated'] = 'lightgray'

E(net)$lty = 1
E(net)[E(net)$Validation=='non-validated']$lty <- 2

set.seed(18)
LO <- layout_as_bipartite(net)
plot(net, 
     vertex.frame.width = 0.5,
     vertex.frame.color = vertex_fcol,
     vertex.color = vertex_col,
     vertex.size = vertex_size,
     vertex.shape = c('square', 'circle')[1+(V(net)$Type_name=='Cell')],
     vertex.label = V(net)$Node_name,
     vertex.label.color = 'black',
     vertex.label.font = 1.0,
     vertex.label.font = c(3, 2)[1+(V(net)$node_type=='Cell')],
     vertex.label.cex = vertex_cex,
     edge.width = edge_width,
     edge.color = edge_color,
     edge.curved = 0.1,
     asp=2.5,
     layout=LO[, 2:1])

### ADD LEGEND
legend(x=1.51, y=-0.71,
       legend=c('Cell Lines', 'Signaling Pathways'), pch=c(21,22), 
       pt.bg=c('lightblue', 'gold'), pt.cex=1.5, cex=0.6, bty='n')

legend(x=1.50, y=-0.79,
       legend=c('Validated', 'Non-validated'),
       col=c('gray', 'lightgray'), lwd=c(2.5,1), cex=0.6, lty=c(1,2), bty='n')




plot(net,
     vertex.frame.width = 1,
     vertex.frame.color = vertex_fcol,
     vertex.color = vertex_col,
     vertex.size = vertex_size,
     vertex.shape = c('square', 'circle')[1+(V(net)$Type_name=='Cell')],
     vertex.label = V(net)$Node_name,
     vertex.label.color = 'black',
     vertex.label.cex = vertex_cex,
     edge.width = edge_width,
     edge.color = edge_color,
     edge.curved = 0.2,
     layout=layout_with_gem)

# layout=layout_with_graphopt
# layout=layout_with_sugiyama
# layout=layout_with_lgl
# layout = layout.random
# layout=layout_nicely
# layout=layout_as_tree
# layout_with_kk
# layout=layout_with_dh
# layout=layout_with_gem



