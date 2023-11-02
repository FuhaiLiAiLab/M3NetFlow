library(shiny)
library(dplyr)
library(igraph)
library(networkD3)
library(kdensity)
library(ggplot2)
library(Jmisc)

round_df <- function(df, digits) {
  nums <- vapply(df, is.numeric, FUN.VALUE = logical(1))
  df[,nums] <- round(df[,nums], digits = digits)
  (df)
}

ui <- fluidPage(
  # titlePanel('Whole Network Interaction'),
  sidebarLayout(
    sidebarPanel(
      selectInput('cell_num', label = 'Selection of Cell Line Name', 
                  choices = list('A498' = 1, 'A549/ATCC' = 2,'ACHN' = 3, 'BT-549' = 4,
                                 'CAKI-1' = 5, 'DU-145' = 6, 'EKVX' = 7, 'HCT-116' = 8, 'HCT-15' = 9,
                                 'HOP-62' = 10,  'HOP-92' = 11, 'HS 578T' = 12, 'IGROV1' = 13, 'K-562' = 14,
                                 'KM12' = 15, 'LOX IMVI' = 16, 'MCF7' = 17, 'MDA-MB-231' = 18,
                                 'MDA-MB-468' = 19, 'NCI-H23' = 20, 'NCI-H460' = 21, 'NCI-H522' = 22,
                                 'OVCAR-3' = 23, 'OVCAR-4' = 24, 'OVCAR-8' = 25, 'PC-3' = 26,
                                 'RPMI-8226' = 27, 'SF-268' = 28, 'SF-295' = 29, 'SF-539' = 30,
                                 'SK-MEL-28' = 32, 'SK-MEL-5' = 33, 'SK-OV-3' = 34, 'SNB-75' = 35,
                                 'SR' = 36, 'SW-620' = 37, 'T-47D' = 38, 'U251' = 39,
                                 'UACC-257' = 40, 'UACC-62' = 41, 'UO-31'= 42)),
      
      selectInput('top_last_num', label = 'Selection of top and last scores drug-gene interaction', 
                  choices = list('Top and Last 1' = 1, 
                                 'Top and Last 2' = 2, 
                                 'Top and Last 3' = 3, 
                                 'Top and Last 4' = 4, 
                                 'Top and Last 5' = 5,
                                 'Top and Last 6' = 6, 
                                 'Top and Last 7' = 7, 
                                 'Top and Last 8' = 8, 
                                 'Top and Last 9' = 9,
                                 'Top and Last 10' = 10),
                  selected = 1),
      
      selectInput('spath', label = 'Selection of specific signaling pathway', 
                  choices = list('Choosing no signaling pathway' = 0,
                                 'AGE-RAGE signaling pathway in diabetic complications' = 1, 
                                 'AMPK signaling pathway' = 2, 
                                 'Adipocytokine signaling pathway' = 3, 
                                 'Apelin signaling pathway' = 4, 
                                 'B cell receptor signaling pathway' = 5,
                                 'C-type lectin receptor signaling pathway' = 6, 
                                 'Calcium signaling pathway' = 7, 
                                 'Chemokine signaling pathway' = 8, 
                                 'ErbB signaling pathway' = 9,
                                 'Estrogen signaling pathway' = 10,
                                 'Fc epsilon RI signaling pathway' = 11, 
                                 'FoxO signaling pathway' = 12, 
                                 'Glucagon signaling pathway' = 13, 
                                 'GnRH signaling pathway' = 14, 
                                 'HIF-1 signaling pathway' = 15,
                                 'Hedgehog signaling pathway' = 16, 
                                 'Hippo signaling pathway' = 17, 
                                 'Hippo signaling pathway - multiple species' = 18, 
                                 'IL-17 signaling pathway' = 19,
                                 'Insulin signaling pathway' = 20,
                                 'JAK-STAT signaling pathway' = 21, 
                                 'MAPK signaling pathway' = 22, 
                                 'NF-kappa B signaling pathway' = 23, 
                                 'NOD-like receptor signaling pathway' = 24, 
                                 'Neurotrophin signaling pathway' = 25,
                                 'Notch signaling pathway' = 26, 
                                 'Oxytocin signaling pathway' = 27, 
                                 'PI3K-Akt signaling pathway' = 28, 
                                 'PPAR signaling pathway' = 29,
                                 'Phospholipase D signaling pathway' = 30,
                                 'Prolactin signaling pathway' = 31, 
                                 'RIG-I-like receptor signaling pathway' = 32, 
                                 'Rap1 signaling pathway' = 33, 
                                 'Ras signaling pathway' = 34, 
                                 'Relaxin signaling pathway' = 35,
                                 'Signaling pathways regulating pluripotency of stem cells' = 36, 
                                 'Sphingolipid signaling pathway' = 37, 
                                 'T cell receptor signaling pathway' = 38, 
                                 'TGF-beta signaling pathway' = 39,
                                 'TNF signaling pathway' = 40,
                                 'Thyroid hormone signaling pathway' = 41, 
                                 'Toll-like receptor signaling pathway' = 42, 
                                 'VEGF signaling pathway' = 43, 
                                 'Wnt signaling pathway' = 44, 
                                 'cAMP signaling pathway' = 45,
                                 'cGMP-PKG signaling pathway' = 46, 
                                 'mTOR signaling pathway' = 47, 
                                 'p53 signaling pathway' = 48),
                  selected = 0),
      
      sliderInput('edge_threshold',
                  'Select the threshold of edge weight to plot',
                  min = 0.1, max = 0.5,
                  value = 0.31),
      
      sliderInput('node_threshold',
                  'Select the threshold of marking important genes',
                  min = 15.0, max = 50.0,
                  value = 22),
      
      sliderInput('giant_comp_threshold',
                  'Select the threshold of each component',
                  min = 0.0, max = 50.0,
                  value = 20.0),
      
      sliderInput('drug_edge_width',
                  'Select the drug-gene edge width',
                  min = 0, max = 5,
                  value = 3.0),
      sliderInput('gene_node_size',
                  'Select the gene node size',
                  min = 0, max = 10,
                  value = 4.0),
      sliderInput('imgene_node_size',
                  'Select the important gene node size',
                  min = 0, max = 10,
                  value = 7.0),
      sliderInput('drug_node_size',
                  'Select the drug node size',
                  min = 0, max = 10,
                  value = 5.0),
      sliderInput('drug_label_size',
                  'Select the label size of drug nodes',
                  min = 0.0001, max = 1.5,
                  value = 0.8),
      sliderInput('gene_label_size',
                  'Select the label size of gene nodes',
                  min = 0.2, max = 1.0,
                  value = 0.6),
      sliderInput('imgene_label_size',
                  'Select the label size of important genes',
                  min = 0.4, max = 1.5,
                  value = 0.8),
    ),
    mainPanel(
      plotOutput(outputId = 'network', height = 1150, width = 1150),
      plotOutput(outputId = 'kde', height = 200, width = 1150)
    )
  )
)

server <- function(input, output) {
  edge_threshold <- reactive({
    input$edge_threshold
  })
  node_threshold <- reactive({
    input$node_threshold
  })
  giant_comp_threshold <- reactive({
    input$giant_comp_threshold
  })
  output$network <- renderPlot({
    ### 1. READ GRAPH [edge_index, node] FROM FILES
    print(input$cell_num)
    fold_n = 0
    fold_number = paste('./analysis-nci/fold_', as.character(fold_n), sep='')
    fold_path = paste(fold_number, '_cell/cell', sep='')
    cell_path = paste(fold_path, as.character(input$cell_num), sep='')
    # cell_path_csv = paste(cell_path, '_undirected.csv', sep='')
    cell_path_csv = paste(cell_path, '.csv', sep='')
    net_edge_weight = read.csv(cell_path_csv)
    hop_net_edge = filter(net_edge_weight, Hop == 'hop1')
    sp_net_edge = filter(net_edge_weight, SpNotation == 'sp1')
    sp_hop_net_edge = filter(sp_net_edge, Hop == 'hop1')
    simple_sp_hop_net_edge = sp_hop_net_edge[, c('From', 'To', 'Attention')]
    net_node = read.csv('./analysis-nci/node_num_dict.csv') # NODE LABEL
    
    ### 2.1 FILTER EDGE BY [edge_weight]
    filter_net_edge = filter(hop_net_edge, Attention > edge_threshold())
    filter_net_edge_node = unique(c(filter_net_edge$From, filter_net_edge$To))
    filter_net_node = net_node[net_node$node_num %in% filter_net_edge_node, ]
    
    ### 2.2 FILTER WITH GIANT COMPONENT
    tmp_net = graph_from_data_frame(d=filter_net_edge, vertices=filter_net_node, directed=F)
    # tmp_net = graph_from_data_frame(d=hop_net_edge, vertices=net_node, directed=F)
    all_components = groups(components(tmp_net))
    # COLLECT ALL LARGE COMPONENTS
    giant_comp_node = c()
    for (x in 1:length(all_components)){
      each_comp = all_components[[x]]
      if (length(each_comp) >= giant_comp_threshold()){
        giant_comp_node = c(giant_comp_node, each_comp)
      }
    }
    
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
    ### SELECT CERTAIN TOP AND LAST ROWS
    ### 2.3.1 ADD FILTERED DRUGS [TOP LAST SCORE DRUGS]
    fold_number = paste('./analysis-nci/fold_', as.character(fold_n), sep='')
    fold_path = paste(fold_number, '_cell/cell', sep='')
    cell_path = paste(fold_path, as.character(input$cell_num), sep='')
    cell_path_drug = paste(cell_path, '_whole_input_idf.csv', sep='')
    cell_path_drug_df = read.csv(cell_path_drug)
    ### SELECT CERTAIN TOP AND LAST ROWS
    # Top Drugs
    print(input$top_last_num)
    top_last_num = input$top_last_num
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
    
    # SELECT FOR CERTAIN DATA POINTS
    refilter_net_drug_node = bind_rows(refilter_net_node, filter_drug_node)
    
    ### 3. BUILD UP GRAPH
    # net = graph_from_data_frame(d=refilter_net_drug_edge, vertices=refilter_net_drug_node, directed=F)
    # 3.1 DESCRIBE NODE DISTRIBUTION
    fold_number = paste('./analysis-nci/fold_', as.character(fold_n), sep='')
    # fold_cell_path = paste(fold_number, '_cell/all_cell_whole_att_deg.csv', sep='')
    fold_cell_path = paste(fold_number, '_cell/all_cell_whole_idf_att_deg_save.csv', sep='')
    all_cell_whole_att_deg_df = read.csv(fold_cell_path)
    cell_whole_att_deg_df = all_cell_whole_att_deg_df[, c(1, as.numeric(input$cell_num)+1)]
    colnames(cell_whole_att_deg_df)[2] <- 'Att_Deg'
    
    cell_filter_drug_node = addCol(filter_drug_node, value=c(Att_Deg=15.0))
    cell_filter_drug_node = cell_filter_drug_node[, c('node_name', 'Att_Deg')]
    colnames(cell_filter_drug_node)[1] <- 'All_gene_num'
    cell_whole_att_deg_df = rbind(cell_whole_att_deg_df, cell_filter_drug_node)
    
    cell_whole_att_deg_df = merge(x = refilter_net_drug_node, y = cell_whole_att_deg_df, by.x = c('node_name'), by.y =c("All_gene_num"))
    cell_whole_att_deg_df = cell_whole_att_deg_df[order(cell_whole_att_deg_df$node_num),]
    cell_whole_att_deg_df <- cell_whole_att_deg_df[, c("node_num", "node_name", "node_type", 'Att_Deg')]
    # 3.2 USE REWEIGHTED IDF NODE DEGREE AS NODE DEGREE
    net = graph_from_data_frame(d=refilter_net_drug_edge, vertices=cell_whole_att_deg_df, directed=F)
    filtered_cell_whole_att_deg_df = filter(cell_whole_att_deg_df, Att_Deg > node_threshold())
    
    # refilter_net_node$att_deg = strength(net, weights=E(net)$Attention)
    ### 4. NETWORK PARAMETERS SETTINGS
    # vertex frame color
    vertex_fcol = rep('black', vcount(net))
    # vertex color
    vertex_col = rep('lightblue', vcount(net))
    vertex_col[V(net)$Att_Deg>=node_threshold()] = 'tomato'
    vertex_col[V(net)$node_type=='drug'] = 'white'
    # vertex size
    vertex_size = rep(input$gene_node_size, vcount(net))
    vertex_size[V(net)$Att_Deg>=node_threshold()] = input$imgene_node_size
    vertex_size[V(net)$node_type=='drug'] = input$drug_node_size
    # vertex cex
    vertex_cex = rep(input$gene_label_size, vcount(net))
    vertex_cex[V(net)$Att_Deg>=node_threshold()] = input$imgene_label_size
    vertex_cex[V(net)$node_type=='drug'] = input$drug_label_size
    # edge width
    edge_width = (E(net)$Attention)*(2.0)
    edge_width[E(net)$edge_type=='Drug-Gene'] = input$drug_edge_width
    # edge color
    edge_color = rep('gray', ecount(net))
    edge_color[E(net)$edge_type=='Drug-Gene'] = 'gold'
    
    ### Signaling Pathway
    print(input$spath)
    spath_notation = paste('sp', input$spath, sep='')
    edge_width[E(net)$SpNotation==spath_notation] = 5
    edge_color[E(net)$SpNotation==spath_notation] = 'black'
    
    ### TOP and LOW DRUGS
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
    
    
    for (spath in 1:48){
      tmp_spath_notation = paste('sp', spath, sep='')
      sp_edges <- as.vector(E(net)$SpNotation==tmp_spath_notation)
      num_all_edges = length(sp_edges)
      if (sum(sp_edges)>(num_all_edges*0.05)){
        print(tmp_spath_notation)
        print(sum(sp_edges))
      }
    }
    print(num_all_edges)
    
    set.seed(18)
    plot(net,
         vertex.frame.width = 2,
         vertex.frame.color = vertex_fcol,
         vertex.color = vertex_col,
         vertex.size = vertex_size,
         vertex.shape = c('square', 'circle')[1+(V(net)$node_type=='gene')],
         vertex.label = V(net)$node_name,
         vertex.label.color = 'black',
         vertex.label.cex = vertex_cex,
         edge.width = edge_width,
         edge.color = edge_color,
         edge.curved = 0.2,
         layout=layout_with_graphopt)
    ### ADD LEGEND
    legend(x=-1.05, y=1.10, # y= -0.72,
           legend=c('Genes', 'Important Genes', 'Drugs'), pch=c(21,21,22), 
           pt.bg=c('lightblue', 'tomato', 'white'), pt.cex=2, cex=1.2, bty='n')
    legend(x=-1.06, y=0.98, # y= -0.85, 
           legend=c('Drug-Gene', 'Top Drug-Gene', 'Low Drug-Gene', 'Gene-Gene', 'Selected Signaling Pathway'),
           col=c('gold', 'purple', 'green', 'gray', 'black'), lwd=c(5,5), cex=1.2, bty='n')
  })
}

# layout=layout_with_graphopt
# layout=layout_with_sugiyama
# layout=layout_with_lgl
# layout = layout.random
# layout=layout_nicely
# layout=layout_as_tree
# layout_with_kk
# layout=layout_with_dh
# layout=layout_with_gem

shinyApp(ui = ui, server = server)


