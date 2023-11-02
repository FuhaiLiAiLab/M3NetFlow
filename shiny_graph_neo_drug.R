library(shiny)
library(dplyr)
library(igraph)
library(networkD3)
library(kdensity)
library(ggplot2)
library(Jmisc)

cell_line_specific_net <- function(fold_n, cell_num, edge_threshold, node_threshold, giant_comp_threshold, top_last_num, layout){
  ### 1. READ GRAPH [edge_index, node] FROM FILES
  fold_number = paste('./analysis-nci/fold_', as.character(fold_n), sep='')
  fold_path = paste(fold_number, '_cell/cell', sep='')
  cell_path = paste(fold_path, as.character(cell_num), sep='')
  # cell_path_csv = paste(cell_path, '_undirected.csv', sep='')
  cell_path_csv = paste(cell_path, '.csv', sep='')
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
  # PLOT EDGE WEIGHT DENSITY DISTRIBUTION
  ggplot(hop_net_edge, aes(x=Attention))+ 
    geom_density(color="darkblue", fill="lightblue")+
    geom_vline(xintercept = density(hop_net_edge$Attention)$Attention[max])+
    scale_x_continuous(breaks=c(0.001, 0.01, 0.1, 0.5, 1.0), trans='log2')+
    geom_vline(aes(xintercept = edge_threshold, color='edge_attention_threshold'), linetype='dashed')+
    # scale_color_manual(values = c("edge_weight_threshold" = "red", 'threshold=0.1'='black'))+
    xlab('Log10 Edge Weight')+
    ylab('Density')+
    ggtitle('KDE Plot of Node Attention Degree')+
    theme(plot.title = element_text(hjust = 1.0, size = 20, face = "bold"), axis.text.x = element_text(angle = 45, hjust=1))
  
  ### 2.2 FILTER WITH GIANT COMPONENT
  tmp_net = graph_from_data_frame(d=filter_net_edge, vertices=filter_net_node, directed=F)
  # tmp_net = graph_from_data_frame(d=hop_net_edge, vertices=net_node, directed=F)
  all_components = groups(components(tmp_net))
  # COLLECT ALL LARGE COMPONENTS
  giant_comp_node = c()
  for (x in 1:length(all_components)){
    each_comp = all_components[[x]]
    if (length(each_comp) >= giant_comp_threshold){
      giant_comp_node = c(giant_comp_node, each_comp)
    }
  }
  # giant_comp_threshold1 = 90
  # giant_comp_threshold2 = 1000
  # for (x in 1:length(all_components)){
  #   each_comp = all_components[[x]]
  #   if (length(each_comp) <= giant_comp_threshold2 & length(each_comp)>=giant_comp_threshold1){
  #     giant_comp_node = c(giant_comp_node, each_comp)
  #   }
  # }
  refilter_net_edge<-subset(filter_net_edge, (From %in% giant_comp_node | To %in% giant_comp_node))
  refilter_net_edge_node = unique(c(refilter_net_edge$From, refilter_net_edge$To))
  refilter_net_node = filter_net_node[filter_net_node$node_num %in% refilter_net_edge_node,]
  # refilter_net_edge<-subset(hop_net_edge, (From %in% giant_comp_node | To %in% giant_comp_node))
  # refilter_net_edge_node = unique(c(refilter_net_edge$From, refilter_net_edge$To))
  # refilter_net_node = net_node[net_node$node_num %in% refilter_net_edge_node,]
  
  ### 2.3 ADD DRUG
  drug_target_df = read.csv('./datainfo-nci/filtered_data/final_drugbank_num.csv')
  drug_target_df = subset(drug_target_df, (Target %in% giant_comp_node))
  colnames(drug_target_df)[1] <- "From"
  colnames(drug_target_df)[2] <- "To"
  drug_target_df = addCol(drug_target_df, value=c(Attention=1.0, 
                                                  Hop='hop1',
                                                  SignalingPath='NoPath',
                                                  SpNotation='NoNotation',
                                                  Cell_Line_Name=refilter_net_edge$Cell_Line_Name[1],
                                                  edge_type='Drug-Gene'))
  drug_target_df = transform(drug_target_df, Attention = as.numeric(Attention))
  refilter_net_edge = addCol(refilter_net_edge, value=c(edge_type='Gene-Gene'))
  refilter_net_drug_edge = bind_rows(refilter_net_edge, drug_target_df)
  # ADD DRUG NODES
  filter_drug_node = net_node[net_node$node_num %in% drug_target_df$From, ]
  
  ### 2.3.1 ADD FILTERED DRUGS [TOP LAST SCORE DRUGS]
  fold_number = paste('./analysis-nci/fold_', as.character(fold_n), sep='')
  fold_path = paste(fold_number, '_cell/cell', sep='')
  cell_path = paste(fold_path, as.character(cell_num), sep='')
  cell_path_drug = paste(cell_path, '_whole_input_idf.csv', sep='')
  cell_path_drug_df = read.csv(cell_path_drug)
  ### SELECT CERTAIN TOP AND LAST ROWS
  # Top Drugs
  top_drugA = cell_path_drug_df[top_last_num, 1]
  top_drugA_num = net_node[net_node$node_name==top_drugA,][[1]]
  top_drugA_ingraph = top_drugA_num %in% drug_target_df$From
  top_drugB = cell_path_drug_df[top_last_num, 4]
  top_drugB_num = net_node[net_node$node_name==top_drugB,][[1]]
  top_drugB_ingraph = top_drugB_num %in% drug_target_df$From
  # Tail Drugs
  low_drugA = head(tail(cell_path_drug_df, n=top_last_num), n=1)[[1]]
  low_drugA_num = net_node[net_node$node_name==low_drugA,][[1]]
  low_drugA_ingraph = low_drugA_num %in% drug_target_df$From
  low_drugB = head(tail(cell_path_drug_df, n=top_last_num), n=1)[[4]]
  low_drugB_num = net_node[net_node$node_name==low_drugB,][[1]]
  low_drugB_ingraph = low_drugB_num %in% drug_target_df$From
  # Top Score & Low Score

  
  
  # SELECT FOR CERTAIN DATA POINTS
  refilter_net_drug_node = bind_rows(refilter_net_node, filter_drug_node)
  
  ### 3. BUILD UP GRAPH
  # net = graph_from_data_frame(d=refilter_net_drug_edge, vertices=refilter_net_drug_node, directed=F)
  # 3.1 DESCRIBE NODE DISTRIBUTION
  fold_number = paste('./analysis-nci/fold_', as.character(fold_n), sep='')
  fold_cell_path = paste(fold_number, '_cell/all_cell_whole_att_deg.csv', sep='')
  all_cell_whole_att_deg_df = read.csv(fold_cell_path)
  cell_whole_att_deg_df = all_cell_whole_att_deg_df[, c(1, cell_num+1)]
  colnames(cell_whole_att_deg_df)[2] <- 'Att_Deg'
  
  cell_filter_drug_node = addCol(filter_drug_node, value=c(Att_Deg=5.0))
  cell_filter_drug_node = cell_filter_drug_node[, c('node_name', 'Att_Deg')]
  colnames(cell_filter_drug_node)[1] <- 'All_gene_num'
  cell_whole_att_deg_df = rbind(cell_whole_att_deg_df, cell_filter_drug_node)
  
  cell_whole_att_deg_df = merge(x = refilter_net_drug_node, y = cell_whole_att_deg_df, by.x = c('node_name'), by.y =c("All_gene_num"))
  cell_whole_att_deg_df = cell_whole_att_deg_df[order(cell_whole_att_deg_df$node_num),]
  cell_whole_att_deg_df <- cell_whole_att_deg_df[, c("node_num", "node_name", "node_type", 'Att_Deg')]
  rownames(cell_whole_att_deg_df) <- cell_whole_att_deg_df$node_num
  # 3.2 USE REWEIGHTED IDF NODE DEGREE AS NODE DEGREE
  net = graph_from_data_frame(d=refilter_net_drug_edge, vertices=cell_whole_att_deg_df, directed=F)
  
  filtered_cell_whole_att_deg_df = filter(cell_whole_att_deg_df, Att_Deg > node_threshold)
  
  ggplot(cell_whole_att_deg_df, aes(x=Att_Deg))+ 
    geom_density(color="darkblue", fill="lightblue")+
    geom_vline(xintercept = density(cell_whole_att_deg_df$Att_Deg)$Att_Deg[max])+
    scale_x_continuous(breaks=c(0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0), trans='log2')+
    geom_vline(aes(xintercept = node_threshold, color='edge_attention_threshold'), linetype='dashed')+
    xlab('Log10 Edge Weight')+
    ylab('Density')+
    ggtitle('KDE Plot of Node Attention Degree')+
    theme(plot.title = element_text(hjust = 1.0, size = 20, face = "bold"), axis.text.x = element_text(angle = 45, hjust=1))
  
  # print(nrow(refilter_net_drug_node))
  # print(nrow(refilter_net_drug_edge))
  
  # refilter_net_node$att_deg = strength(net, weights=E(net)$Attention)
  ### 4. NETWORK PARAMETERS SETTINGS
  # vertex frame color
  vertex_fcol = rep('black', vcount(net))
  # vertex color
  vertex_col = rep('lightblue', vcount(net))
  vertex_col[V(net)$Att_Deg>=node_threshold] = 'tomato'
  vertex_col[V(net)$node_type=='drug'] = 'white'
  # vertex size
  vertex_size = rep(4, vcount(net))
  vertex_size[V(net)$Att_Deg>=node_threshold] = 7
  vertex_size[V(net)$node_type=='drug'] = 5
  # vertex cex
  vertex_cex = rep(0.6, vcount(net))
  vertex_cex[V(net)$Att_Deg>=node_threshold] = 0.8
  vertex_cex[V(net)$node_type=='drug'] = 0.8
  # edge width
  edge_width = (E(net)$Attention)*(2)
  edge_width[E(net)$edge_type=='Drug-Gene'] =  3
  # edge color
  edge_color = rep('gray', ecount(net))
  edge_color[E(net)$edge_type=='Drug-Gene'] = 'gold'
  # edge_color[E(net)$SpNotation=='sp9'] = 'black'
  
  if (top_drugA_ingraph){
    edge_width[E(net)[from(V(net)[as.character(top_drugA_num)])]]=5
    edge_color[E(net)[from(V(net)[as.character(top_drugA_num)])]]='purple'
  }
  if (top_drugB_ingraph){
    edge_width[E(net)[from(V(net)[as.character(top_drugB_num)])]]=5
    edge_color[E(net)[from(V(net)[as.character(top_drugB_num)])]]='purple'
  }
  if (low_drugA_ingraph){
    edge_width[E(net)[from(V(net)[as.character(low_drugA_num)])]]=5
    edge_color[E(net)[from(V(net)[as.character(low_drugA_num)])]]='green'
  }
  if (low_drugB_ingraph){
    edge_width[E(net)[from(V(net)[as.character(low_drugB_num)])]]=5
    edge_color[E(net)[from(V(net)[as.character(low_drugB_num)])]]='green'
  }
  
  E_gene_net = E(net)[E(net)$edge_type=='Gene-Gene']
  E_gene_net_edge_sum = sum(E_gene_net$Attention)
  
  cell_num_list = c()
  tmp_spath_notation_list = c()
  tmp_sp_edges_list = c()
  tmp_all_edges_list = c()
  for (spath in 1:48){
    tmp_spath_notation = paste('sp', spath, sep='')
    sp_edges <- E_gene_net[E_gene_net$SpNotation==tmp_spath_notation]
    sp_edges_sum <- sum(sp_edges$Attention)
    if (sp_edges_sum>(E_gene_net_edge_sum*0.05)){
      print(tmp_spath_notation)
      tmp_spath_notation_list = c(tmp_spath_notation_list, tmp_spath_notation)
      print(sp_edges_sum)
      tmp_sp_edges_list = c(tmp_sp_edges_list, sp_edges_sum)
      cell_num_list = c(cell_num_list, cell_num)
      tmp_all_edges_list = c(tmp_all_edges_list, E_gene_net_edge_sum)
    }
  }
  
  # set.seed(18)
  # plot(net,
  #      vertex.frame.width = 2,
  #      vertex.frame.color = vertex_fcol,
  #      vertex.color = vertex_col,
  #      vertex.size = vertex_size,
  #      # vertex.shape = 'circle',
  #      vertex.shape = c('square', 'circle')[1+(V(net)$node_type=='gene')],
  #      vertex.label = V(net)$node_name,
  #      vertex.label.color = 'black',
  #      vertex.label.font = 1.0,
  #      vertex.label.degree= -pi/2,
  #      #vertex.label.font = c(4, 2)[1+(V(net)$node_type=='gene')],
  #      vertex.label.cex = vertex_cex,
  #      edge.width = edge_width,
  #      edge.color = edge_color,
  #      # edge.alpha = c(input$edge_opacity, 1.0)[1+(E(net)$edge_type=='gene-gene')],
  #      edge.curved = 0.2,
  #      layout=layout_with_graphopt)
  
  return(c(cell_num_list, tmp_spath_notation_list, tmp_sp_edges_list, tmp_all_edges_list))
}

percentile_df <- read.csv('./analysis-nci/percentile.csv')

cell_sp_matrix <- matrix(1, nrow = 1, ncol = 4)
for(i in 1:nrow(percentile_df)) {       
  cell_num = percentile_df[i,3]
  gene_percentile = percentile_df[i,4]
  edge_percentile = percentile_df[i,5]
  cell_sp <- cell_line_specific_net(fold_n=0, cell_num=cell_num, 
                         edge_threshold=edge_percentile, node_threshold=gene_percentile, 
                         giant_comp_threshold=20, top_last_num=1, layout=layout_with_graphopt)
  cell_sp <- t(matrix(cell_sp, nrow=4, byrow=TRUE))
  cell_sp_matrix <- rbind(cell_sp_matrix, cell_sp)
  print(cell_sp)
}

cell_sp_df <- as.data.frame(cell_sp_matrix)
colnames(cell_sp_df) <- c("cell_num", "sp_notation", "sp_num_edges", "all_num_edges")
cell_sp_df <- cell_sp_df[-1,]
rownames(cell_sp_df) <- NULL

cancer_cell_line_df <- read.csv('./datainfo-nci/filtered_data/cell_line_cancer_name_map_dict.csv')
kegg_sp_df <- read.csv('./datainfo-nci/filtered_data/kegg_sp_map.csv')
cancer_cell_sp_df <- merge(cell_sp_df, cancer_cell_line_df, by.x = 'cell_num', by.y ='Cell_Line_Num')
cancer_cell_sp_df <- merge(cancer_cell_sp_df, kegg_sp_df, by.x = 'sp_notation', by.y ='SpNotation')
cancer_cell_sp_df <- cancer_cell_sp_df[, c('Cancer_name', 'Cell_Line_Name', 'SignalingPath', 'sp_num_edges', 'all_num_edges')]

cancer_cell_sp_df <- arrange(cancer_cell_sp_df, Cancer_name, Cell_Line_Name, SignalingPath)
write.csv(cancer_cell_sp_df, './analysis-nci/cancer_cell_sp.csv', row.names = FALSE)

# layout=layout_with_graphopt
# layout=layout_with_sugiyama
# layout=layout_with_lgl
# layout = layout.random
# layout=layout_nicely
# layout=layout_as_tree
# layout_with_kk
# layout=layout_with_dh
# layout=layout_with_gem