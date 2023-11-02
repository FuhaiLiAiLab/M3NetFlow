library(ggplot2)
library(reshape2)
library(datasets)
library(ggpubr)
library(gridExtra)

round_df <- function(df, digits) {
  nums <- vapply(df, is.numeric, FUN.VALUE = logical(1))
  df[,nums] <- round(df[,nums], digits = digits)
  (df)
}

cell_line_cancer_name_map_df = read.csv('./datainfo-nci/filtered_data/cell_line_cancer_name_map_dict.csv')

cell_whole_idf_com_drugAB_boxplot <- function(top_last_num, fold_n, cell_num){
  cancer_name = cell_line_cancer_name_map_df[cell_line_cancer_name_map_df$Cell_Line_Num==cell_num,][[3]]
  cell_name = cell_line_cancer_name_map_df[cell_line_cancer_name_map_df$Cell_Line_Num==cell_num,][[1]]
  plot_name3 = paste(cell_name, 'Highest and Lowest Drugs\' Targeted Genes Degree Distributions', sep=' ')
  setwd('/Users/muhaha/Files/VS-Files/Combo-GeoSubSig-Test')
  fold_number = paste('./analysis-nci/fold_', as.character(fold_n), sep='')
  fold_path = paste(fold_number, '_cell/cell', sep='')
  print(fold_path)
  cell_path = paste(fold_path, as.character(cell_num), sep='')
  cell_path_drugA = paste(cell_path, '_drugA_whole_input_idf.csv', sep='')
  cell_drugA_df = read.csv(cell_path_drugA)
  print(cell_path_drugA)
  cell_drugA_df$Drug_type = c('drugA')
  cell_drugA_df = round_df(cell_drugA_df, digits=3)
  cell_path_drugB = paste(cell_path, '_drugB_whole_input_idf.csv', sep='')
  cell_drugB_df = read.csv(cell_path_drugB)
  print(cell_path_drugB)
  cell_drugB_df$Drug_type = c('drugB')
  cell_drugB_df = round_df(cell_drugB_df, digits=3)
  cell_drug_df <- rbind(cell_drugA_df, cell_drugB_df)
  # Level Score As Factor
  allLevels <- levels(factor(c(cell_drug_df$Score)))
  cell_drug_df$Score <- factor(cell_drug_df$Score,levels=(allLevels))
  # Top Score & Low Score
  top_last_num = 5
  score = unique(cell_drug_df$Score)
  top_score = as.vector(head(score, top_last_num))
  low_score = as.vector(tail(score, top_last_num))
  top_filter_cell_drug_df = cell_drug_df[cell_drug_df$Score %in% top_score, ]
  low_filter_cell_drug_df = cell_drug_df[cell_drug_df$Score %in% low_score, ]
  
  #####################
  # COMBINE Drug A & DrugB
  cell_drug_df$Drug_type = c('drug')
  top_last_num = 5
  score = unique(cell_drug_df$Score)
  top_score = as.vector(head(score, top_last_num))
  low_score = as.vector(tail(score, top_last_num))
  top_filter_cell_drug_df = cell_drug_df[cell_drug_df$Score %in% top_score, ]
  low_filter_cell_drug_df = cell_drug_df[cell_drug_df$Score %in% low_score, ]
  
  # COMBINE TOP LOW
  top_filter_cell_drug_df$Drug_type = c('Top')
  low_filter_cell_drug_df$Drug_type = c('Low')
  combine_filter_cell_drug_df =rbind(top_filter_cell_drug_df, low_filter_cell_drug_df)
  
  res.ftest <- var.test(att_deg ~ Drug_type, data = combine_filter_cell_drug_df)
  print(res.ftest)
  ftest_value = res.ftest
  
  res.ttest <- t.test(att_deg ~ Drug_type, data = combine_filter_cell_drug_df)
  # print(res.ttest)
  # ttest_value = res.ttest
  
  top_mean = mean(top_filter_cell_drug_df$att_deg)
  low_mean = mean(low_filter_cell_drug_df$att_deg)
  print(top_mean)
  print(low_mean)
  
  top_median = median(top_filter_cell_drug_df$att_deg)
  low_median = median(low_filter_cell_drug_df$att_deg)
  print(top_median)
  print(low_median)
  
  top_target_node_att_deg_list = as.list(top_filter_cell_drug_df$att_deg)
  low_target_node_att_deg_list = as.list(low_filter_cell_drug_df$att_deg)
  vec1 <- unlist(top_target_node_att_deg_list)
  vec2 <- unlist(low_target_node_att_deg_list)
  
  test_result <- t.test(vec1, vec2)
  p_value <- test_result$p.value
  
  # Get the size of the smaller list
  min_length <- min(length(vec1), length(vec2))
  
  # Bootstrap function
  bootstrap_sample <- function(vec, size) {
    sample(vec, size, replace=TRUE)
  }
  
  # Get bootstrapped samples
  vec1_bootstrapped <- bootstrap_sample(vec1, min_length)
  vec2_bootstrapped <- bootstrap_sample(vec2, min_length)
  
  # Combine for plotting
  df <- data.frame(
    Value = c(vec1_bootstrapped, vec2_bootstrapped),
    Group = factor(rep(c("Top", "Low"), each=min_length))
  )
  
  # Extract default ggplot colors
  default_colors <- scale_fill_discrete(aesthetics = "fill")$palette(2)
  
  # Plot
  g1 <- ggplot(df, aes(x=Value, color=Group)) +
    stat_density(aes(y=..density..), geom="line", position="identity") +
    scale_color_manual(values = default_colors) +
    theme_minimal() +
    labs(x = "Node Importance Scores", color=NULL, title=cell_name) +
    theme(axis.line = element_line(color = "black", size = 0.5),
                   axis.ticks = element_line(color = "black"), plot.title = element_text(size = 10, face = "bold")) +  # Place legend inside the plot, adjust as needed
    annotate("text", x = Inf, y = Inf, label = paste("p =", round(p_value, 4)),
             hjust = "right", vjust = "top", size = 4, color = "black")
    # labs(x = "Node Importance Scores", color=NULL) +
    # theme(axis.line = element_line(color = "black", size = 0.5),
    #                axis.ticks = element_line(color = "black"))
  
  
  
  # df1 <- data.frame(value = vec1, Group = "Top")
  # df2 <- data.frame(value = vec2, Group = "Low")
  # combined_df <- rbind(df1, df2)
  # 
  # default_colors <- scale_fill_discrete(aesthetics = "fill")$palette(2)
  # 
  # g1 <- ggplot(data = combined_df, aes(x = value, y = ..density.., color = Group)) + 
  #   stat_density(geom = "line") + 
  #   annotate("text", x = Inf, y = Inf, label = paste("p-value =", round(p_value, 4)), 
  #            vjust = "top", hjust = "right", size = 4, color = "black") +
  #   theme_minimal() +
  #   theme(axis.line = element_line(color = "black", size = 0.5),
  #         axis.ticks = element_line(color = "black")) +
  #   labs(x = "Node Importance Scores") +
  #   # scale_color_manual(values = c("Top" = "pink", "Low" = "skyblue"))
  #   scale_color_manual(values = c("Top" = default_colors[1], "Low" = default_colors[2]))
  
  
  # g1 <- ggplot() + 
  #   stat_density(data = data.frame(x = vec1), aes(x, y=..density..), geom="line", color="pink") + 
  #   stat_density(data = data.frame(x = vec2), aes(x, y=..density..), geom="line", color="skyblue") +
  #   annotate("text", x = Inf, y = Inf, label = paste("p-value =", round(p_value, 4)), 
  #            vjust = "top", hjust = "right", size = 4, color = "black") +
  #   theme_minimal() +
  #   theme(axis.line = element_line(color = "black", size = 0.5),
  #         axis.ticks = element_line(color = "black")) +
  #   labs(x = "Node Importance Scores")
    
  # g1 <- ggplot() + 
  #   geom_density(data = data.frame(x = vec1), aes(x), fill="pink", alpha=0.5) + 
  #   geom_density(data = data.frame(x = vec2), aes(x), fill="skyblue", alpha=0.5) +
  #   theme_minimal() + 
  #   labs(fill = "List")
  
  # bin_width = 2.5  # Adjust this value as required for the bin width
  # g1 <- ggplot() + 
  #   geom_histogram(data = data.frame(x = vec1), aes(x), fill="red", alpha=0.5, bins=30) + 
  #   geom_histogram(data = data.frame(x = vec2), aes(x), fill="blue", alpha=0.5, bins=30) +
  #   theme_minimal() + 
  #   labs(fill = "List")

  cell_number = paste('cell_', as.character(cell_num), sep='')
  ggsave(paste(cell_number, '.pdf', sep=''), width = 6, height = 3.6, g1, path = './analysis-nci2/')
  
  return(c(ftest_value, ttest_value, top_mean, low_mean, top_median, low_median)) 
}

cell_line_stat = cell_whole_idf_com_drugAB_boxplot(top_last_num=5, fold_n=0, cell_num=6)

ftest_value_list = c()
ttest_value_list = c()
top_mean_list = c()
low_mean_list = c()
top_median_list = c()
low_median_list = c()

cell_line_cancer_name_map_df = read.csv('./datainfo-nci/filtered_data/cell_line_cancer_name_map_dict.csv')

for (cell_line in 1:nrow(cell_line_cancer_name_map_df)){
  cell_line_stat = cell_whole_idf_com_drugAB_boxplot(top_last_num=5, fold_n=0, cell_num=cell_line)
  ftest_value = cell_line_stat[3]$p.value
  ttest_value = cell_line_stat[12]$p.value
  top_mean = cell_line_stat[20][[1]]
  low_mean = cell_line_stat[21][[1]]
  top_median = cell_line_stat[22][[1]]
  low_median = cell_line_stat[23][[1]]
  ftest_value_list = c(ftest_value_list, ftest_value)
  ttest_value_list = c(ttest_value_list, ttest_value)
  top_mean_list = c(top_mean_list, top_mean)
  low_mean_list = c(low_mean_list, low_mean)
  top_median_list = c(top_median_list, top_median)
  low_median_list = c(low_median_list, low_median)
}

# 
# cell_line_cancer_name_map_df$f_value = ftest_value_list
# cell_line_cancer_name_map_df$t_value = ttest_value_list
# cell_line_cancer_name_map_df$top_mean = top_mean_list
# cell_line_cancer_name_map_df$low_mean = low_mean_list
# cell_line_cancer_name_map_df$top_median = top_median_list
# cell_line_cancer_name_map_df$low_median = low_median_list
# write.csv(cell_line_cancer_name_map_df, './analysis-nci/top_low_p_value_comparisons.csv')


