library(shiny)
library(dplyr)
library(igraph)
library(networkD3)
library(kdensity)
library(ggplot2)
library(RColorBrewer)

each_cell_search_gene <- function(fold_n, cell_num, edge_threshold, giant_comp_threshold, whole_net, reweight){
  fold_number = paste('./analysis-nci/fold_', as.character(fold_n), sep='')
  if (reweight){
    fold_path = paste(fold_number, '_cell-w/reweight_cell', sep='')
  }else{
    fold_path = paste(fold_number, '_cell/cell', sep='')
  } 
  cell_path = paste(fold_path, as.character(cell_num), sep='')
  # cell_path_csv = paste(cell_path, '_undirected.csv', sep='')
  cell_path_csv = paste(cell_path, '.csv', sep='')
  ### 1. READ GRAPH [edge_index, node] FROM FILES
  setwd('/Users/muhaha/Files/VS-Files/Combo-GeoSubSig-Test')
  net_edge_weight = read.csv(cell_path_csv)
  hop_net_edge = filter(net_edge_weight, Hop == 'hop1')
  sp_net_edge = filter(net_edge_weight, SpNotation == 'sp1')
  sp_hop_net_edge = filter(sp_net_edge, Hop == 'hop1')
  simple_sp_hop_net_edge = sp_hop_net_edge[, c('From', 'To', 'Attention')]
  net_node = read.csv('./analysis-nci/node_num_dict.csv') # NODE LABEL
  
  ### 2.1 FILTER EDGE BY [edge_weight]

  filter_net_edge = filter(hop_net_edge, Attention > edge_threshold)
  filter_net_edge_node = unique(c(filter_net_edge$From, filter_net_edge$To))
  filter_net_node = net_node[net_node$node_num %in% filter_net_edge_node, ]
  ### 2.2 FILTER WITH GIANT COMPONENT
  tmp_net = graph_from_data_frame(d=filter_net_edge, vertices=filter_net_node, directed=F)
  all_components = groups(components(tmp_net))
  # COLLECT ALL LARGE COMPONENTS
  giant_comp_node = c()
  for (x in 1:length(all_components)){
    each_comp = all_components[[x]]
    if (length(each_comp) >= giant_comp_threshold){
      giant_comp_node = c(giant_comp_node, each_comp)
    }
  }
  refilter_net_edge<-subset(filter_net_edge, (From %in% giant_comp_node | To %in% giant_comp_node))
  refilter_net_edge_node = unique(c(refilter_net_edge$From, refilter_net_edge$To))
  refilter_net_node = filter_net_node[filter_net_node$node_num %in% refilter_net_edge_node,]
  ### 3. BUILD UP GRAPH
  net = graph_from_data_frame(d=refilter_net_edge, vertices=refilter_net_node, directed=F)
  refilter_net_node$att_deg = strength(net, weights=E(net)$Attention)
  if (whole_net){
    cell_path_save = paste(cell_path, '_wgc.csv', sep='')
  }
  else{
    cell_path_save = paste(cell_path, '_gc.csv', sep='')
  }
  write.csv(refilter_net_node, cell_path_save, row.names = FALSE)
  print(length(refilter_net_node[[1]]))
}

setwd('/Users/muhaha/Files/VS-Files/Combo-GeoSubSig-Test')
cell_map_dict_df = read.csv('./datainfo-nci/filtered_data/cell_line_map_dict.csv')
fold_n = 0
num_cell = length(cell_map_dict_df[[1]])
for (x in 1:num_cell){
  each_cell_search_gene(fold_n, x, edge_threshold = 0.2, giant_comp_threshold = 20, whole_net = F, reweight=F)
}

# CREATE HEATMAP ACROSS CELL LINES
# all_cell_att_deg_df = read.csv('./analysis-nci/fold_4_cell/all_cell_att_deg.csv')
all_cell_att_deg_df = read.csv('./analysis-nci/fold_0_cell/all_cell_whole_idf_att_deg_save.csv')
rownames(all_cell_att_deg_df) <- all_cell_att_deg_df$All_gene_num
drop <- c('All_gene_num')
all_cell_att_deg_df = all_cell_att_deg_df[,!(names(all_cell_att_deg_df) %in% drop)]
heatmap(data.matrix(all_cell_att_deg_df), 
        # Colv=NA,
        Rowv=NA,
        barScale=1,
        col=brewer.pal(9,"Blues"))

### USE AS HEAT COLORBAR
filled.contour(z=as.matrix(all_cell_att_deg_df),color.palette = colorRampPalette(brewer.pal(9, "Blues")))
legend(x="bottomright", legend=c("min1", "min2", "min3", "ave1","ave2","ave3", "max1", "max2", "max3"),
       fill=colorRampPalette(brewer.pal(9, "Blues"))(9))

# SELECT CANCER CELL LINES
cell_cancer_name_map_dict_df = read.csv('./datainfo-nci/filtered_data/cell_line_cancer_name_map_dict.csv')
filter_cancer_cell = filter(cell_cancer_name_map_dict_df, Cancer_name == 'CNS')
cancer_cell_att_deg_df = all_cell_att_deg_df[, as.vector(filter_cancer_cell$Cell_Line_Num)]

heatmap(data.matrix(cancer_cell_att_deg_df), 
        # Colv=NA,
        # Rowv=NA,
        col=brewer.pal(9,"Blues"))




# # CUT VALUE WITH [att_deg]
# all_cell_att_deg_df = read.csv('./analysis-nci/all_cell_att_deg.csv')
# rownames(all_cell_att_deg_df) <- all_cell_att_deg_df$All_gene_num
# drop <- c('All_gene_num')
# all_cell_att_deg_df = all_cell_att_deg_df[,!(names(all_cell_att_deg_df) %in% drop)]
# all_cell_att_deg_df[all_cell_att_deg_df >= 2] <- 2
# old_all_cell_att_deg_df = all_cell_att_deg_df
# all_cell_att_deg_df[all_cell_att_deg_df < 2] <- 1
# all_cell_att_deg_df[old_all_cell_att_deg_df ==0] <- 0
# heatmap(as.matrix(all_cell_att_deg_df))


