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
      selectInput('type_num', label = 'Selection of Patient Type', 
                  choices = list('AD' = 1, 'Non-AD' = 2)),
      
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
                  min = 0.0, max = 0.25,
                  value = 0.106),
      
      sliderInput('node_threshold',
                  'Select the threshold of marking important genes',
                  min = 1.0, max = 5.0,
                  value = 2),
      
      sliderInput('pvalue_threshold',
                  'Select the threshold of marking important genes by p-values',
                  min = 0, max = 0.3,
                  value = 0.1),
      
      sliderInput('giant_comp_threshold',
                  'Select the threshold of each component',
                  min = 0.0, max = 20.0,
                  value = 15),

      sliderInput('gene_node_size',
                  'Select the gene node size',
                  min = 0, max = 10,
                  value = 5.0),

      sliderInput('imgene_node_size',
                  'Select the important gene node size',
                  min = 0, max = 10,
                  value = 6.0),

      sliderInput('gene_label_size',
                  'Select the label size of gene nodes',
                  min = 0.5, max = 1.0,
                  value = 0.75),

      sliderInput('imgene_label_size',
                  'Select the label size of important genes',
                  min = 0.5, max = 1.5,
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
  pvalue_threshold <- reactive({
    input$pvalue_threshold
  })
  giant_comp_threshold <- reactive({
    input$giant_comp_threshold
  })
  output$network <- renderPlot({
    ### 1. READ GRAPH [edge_index, node] FROM FILES
    print(input$type_num)
    if (input$type_num == 1){
      cell_path = './ROSMAP-analysis/avg_analysis/average_attention_AD'
    }else if (input$type_num == 2){ 
      cell_path = './ROSMAP-analysis/avg_analysis/average_attention_NOAD'
    }
    cell_path_csv = paste(cell_path, '.csv', sep='')
    net_edge_weight = read.csv(cell_path_csv)
    if (input$type_num == 1){
      net_node = read.csv('./ROSMAP-analysis/avg_analysis/map-all-gene-AD-Att_deg_pvalue.csv') # NODE LABEL
    }else if (input$type_num == 2){
      net_node = read.csv('./ROSMAP-analysis/avg_analysis/map-all-gene-NOAD-Att_deg_pvalue.csv') # NODE LABEL
    }
    
    ### 2.1 FILTER EDGE BY [edge_weight]
    filter_net_edge = filter(net_edge_weight, Attention > edge_threshold())
    filter_net_edge_node = unique(c(filter_net_edge$From, filter_net_edge$To))
    filter_net_node = net_node[net_node$Gene_num %in% filter_net_edge_node, ]
    
    ### 2.2 FILTER WITH GIANT COMPONENT
    tmp_net = graph_from_data_frame(d=filter_net_edge, vertices=filter_net_node, directed=F)
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
    refilter_net_node = filter_net_node[filter_net_node$Gene_num %in% refilter_net_edge_node,]
    refilter_node_path = paste(cell_path, '_refilter_node_weight_df.csv', sep='')
    write.csv(refilter_net_node, refilter_node_path, row.names=F)
    
    ### 3. BUILD UP GRAPH
    net = graph_from_data_frame(d=refilter_net_edge, vertices=refilter_net_node, directed=F)

    ### 4. NETWORK PARAMETERS SETTINGS
    # vertex frame color as long as one of the 10 pvalues is smaller than pvalue_threshold, makr it as black 
    # feature includes (['proteomics_pvalues', 'gene-expression_pvalues', 'cnv_del_pvalues', 'cnv_dup_pvalues', 'cnv_mcnv_pvalues', 'methy-Downstream_pvalues', 
    #                     'methy-Core-Promoter_pvalues', 'methy-Proximal-Promoter_pvalues', 'methy-Distal-Promoter_pvalues', 'methy-Upstream_pvalues'])
    vertex_fcol = rep(NA, vcount(net))
    vertex_fcol[V(net)$gene_expression_pvalues < pvalue_threshold()] = '#a569bd'
    # # Combine all conditions into one logical OR condition
    # mark_condition = V(net)$proteomics_pvalues <= pvalue_threshold() |
    #   V(net)$gene_expression_pvalues <= pvalue_threshold() |
    #   V(net)$cnv_del_pvalues <= pvalue_threshold() |
    #   V(net)$cnv_dup_pvalues <= pvalue_threshold() |
    #   V(net)$cnv_mcnv_pvalues <= pvalue_threshold() |
    #   V(net)$methy_Downstream_pvalues <= pvalue_threshold() |
    #   V(net)$methy_Core_Promoter_pvalues <= pvalue_threshold() |
    #   V(net)$methy_Proximal_Promoter_pvalues <= pvalue_threshold() |
    #   V(net)$methy_Distal_Promoter_pvalues <= pvalue_threshold() |
    #   V(net)$methy_Upstream_pvalues <= pvalue_threshold()
    # vertex_fcol[mark_condition] = '#a569bd'

    # vertex color
    vertex_col = rep('lightblue', vcount(net))
    vertex_col[V(net)$Att_deg>=node_threshold()] = 'tomato'
    # vertex size
    vertex_size = rep(input$gene_node_size, vcount(net))
    vertex_size[V(net)$Att_deg>=node_threshold()] = input$imgene_node_size
    # vertex cex
    vertex_cex = rep(input$gene_label_size, vcount(net))
    vertex_cex[V(net)$Att_deg>=node_threshold()] = input$imgene_label_size
    # edge width
    edge_width = (E(net)$Attention)*(7.0)
    # edge color
    edge_color = rep('gray', ecount(net))

    
    ### Signaling Pathway
    print(input$spath)
    spath_notation = paste('sp', input$spath, sep='')
    edge_width[E(net)$SpNotation==spath_notation] = 5
    edge_color[E(net)$SpNotation==spath_notation] = 'black'
    
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
         vertex.frame.width = 4.0,
         vertex.frame.color = vertex_fcol,
         vertex.color = vertex_col,
         vertex.size = vertex_size,
         vertex.shape = 'circle',
         vertex.label = V(net)$Gene_name,
         vertex.label.color = 'black',
         vertex.label.cex = vertex_cex,
         vertex.label.font = 2,
         edge.width = edge_width,
         edge.color = edge_color,
         edge.curved = 0.2,
         layout=layout_with_graphopt)
    ### Add Legend
    legend(x= -1.15, y=1.13, # y= -0.72,
           legend=c('Genes', 'Important Genes Marked By Attention-based Scores', 'Genes With Significant Differences'), pch=c(21,21,21), 
           col=c('lightblue', 'tomato', '#a569bd'),
           pt.bg=c('lightblue', 'tomato', 'white'), 
           pt.lwd=c(1, 1, 4),
           pt.cex=2, cex=1.2, bty='n')
    legend(x= -1.16, y=1.00, # y= -0.85, 
           legend=c('Gene-Gene', 'Selected Signaling Pathway'),
           col=c('gray', 'black'), lwd=c(3,3), cex=1.2, bty='n')
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


