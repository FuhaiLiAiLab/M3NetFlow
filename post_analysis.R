library(BioCircos)
library(dplyr)

setwd('/Users/muhaha/Files/VS-Files/Combo-GeoSubSig-Test')
hop1_edge_filtered_average_join_merge_df = read.csv('./analysis-nci/fold_0_cell/hop1_edge_filtered_average_join_merge.csv')

cancer_types = unique(hop1_edge_filtered_average_join_merge_df$Cancer_name)
hop1_edge_filtered_average_join_merge_df$FactorCancers <- factor(hop1_edge_filtered_average_join_merge_df$Cancer_name, levels = unique(hop1_edge_filtered_average_join_merge_df$Cancer_name))


# sp1
sp1_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp1'))
sp1_df$FactorCancers <- factor(sp1_df$Cancer_name, levels = unique(sp1_df$Cancer_name))
sp1 = sp1_df$Attention
cancer_types_counts <- table(sp1_df$FactorCancers)
cancer_box_positions = unlist(sapply(cancer_types_counts, seq))
cancer_box = rep(cancer_types, cancer_types_counts)
# sp2
sp2_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp2'))
sp2_df$FactorCancers <- factor(sp2_df$Cancer_name, levels = unique(sp2_df$Cancer_name))
sp2 = sp2_df$Attention
# sp3
sp3_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp3'))
sp3_df$FactorCancers <- factor(sp3_df$Cancer_name, levels = unique(sp3_df$Cancer_name))
sp3 = sp3_df$Attention
# sp4
sp4_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp4'))
sp4_df$FactorCancers <- factor(sp4_df$Cancer_name, levels = unique(sp4_df$Cancer_name))
sp4 = sp2_df$Attention
# sp5
sp5_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp5'))
sp5_df$FactorCancers <- factor(sp5_df$Cancer_name, levels = unique(sp5_df$Cancer_name))
sp5 = sp5_df$Attention
# sp6
sp6_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp6'))
sp6_df$FactorCancers <- factor(sp6_df$Cancer_name, levels = unique(sp6_df$Cancer_name))
sp6 = sp6_df$Attention
# sp7
sp7_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp7'))
sp7_df$FactorCancers <- factor(sp7_df$Cancer_name, levels = unique(sp7_df$Cancer_name))
sp7 = sp7_df$Attention
# sp8
sp8_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp8'))
sp8_df$FactorCancers <- factor(sp8_df$Cancer_name, levels = unique(sp8_df$Cancer_name))
sp8 = sp8_df$Attention
# sp9
sp9_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp9'))
sp9_df$FactorCancers <- factor(sp9_df$Cancer_name, levels = unique(sp9_df$Cancer_name))
sp9 = sp9_df$Attention
# sp10
sp10_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp10'))
sp10_df$FactorCancers <- factor(sp10_df$Cancer_name, levels = unique(sp10_df$Cancer_name))
sp10 = sp10_df$Attention
# sp11
sp11_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp11'))
sp11_df$FactorCancers <- factor(sp11_df$Cancer_name, levels = unique(sp11_df$Cancer_name))
sp11 = sp11_df$Attention
# sp12
sp12_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp12'))
sp12_df$FactorCancers <- factor(sp12_df$Cancer_name, levels = unique(sp12_df$Cancer_name))
sp12 = sp12_df$Attention
# sp13
sp13_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp13'))
sp13_df$FactorCancers <- factor(sp13_df$Cancer_name, levels = unique(sp13_df$Cancer_name))
sp13 = sp13_df$Attention
# sp14
sp14_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp14'))
sp14_df$FactorCancers <- factor(sp14_df$Cancer_name, levels = unique(sp14_df$Cancer_name))
sp14 = sp14_df$Attention
# sp15
sp15_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp15'))
sp15_df$FactorCancers <- factor(sp15_df$Cancer_name, levels = unique(sp15_df$Cancer_name))
sp15 = sp15_df$Attention
# sp16
sp16_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp16'))
sp16_df$FactorCancers <- factor(sp16_df$Cancer_name, levels = unique(sp16_df$Cancer_name))
sp16 = sp16_df$Attention
# sp17
sp17_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp17'))
sp17_df$FactorCancers <- factor(sp17_df$Cancer_name, levels = unique(sp17_df$Cancer_name))
sp17 = sp17_df$Attention
# sp18
sp18_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp18'))
sp18_df$FactorCancers <- factor(sp18_df$Cancer_name, levels = unique(sp18_df$Cancer_name))
sp18 = sp18_df$Attention
# sp19
sp19_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp19'))
sp19_df$FactorCancers <- factor(sp19_df$Cancer_name, levels = unique(sp19_df$Cancer_name))
sp19 = sp19_df$Attention
# sp20
sp20_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp20'))
sp20_df$FactorCancers <- factor(sp20_df$Cancer_name, levels = unique(sp20_df$Cancer_name))
sp20 = sp20_df$Attention
# sp21
sp21_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp21'))
sp21_df$FactorCancers <- factor(sp21_df$Cancer_name, levels = unique(sp21_df$Cancer_name))
sp21 = sp21_df$Attention
# sp22
sp22_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp22'))
sp22_df$FactorCancers <- factor(sp22_df$Cancer_name, levels = unique(sp22_df$Cancer_name))
sp22 = sp22_df$Attention
# sp23
sp23_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp23'))
sp23_df$FactorCancers <- factor(sp23_df$Cancer_name, levels = unique(sp23_df$Cancer_name))
sp23 = sp23_df$Attention
# sp24
sp24_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp24'))
sp24_df$FactorCancers <- factor(sp24_df$Cancer_name, levels = unique(sp24_df$Cancer_name))
sp24 = sp24_df$Attention
# sp25
sp25_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp25'))
sp25_df$FactorCancers <- factor(sp25_df$Cancer_name, levels = unique(sp25_df$Cancer_name))
sp25 = sp25_df$Attention
# sp26
sp26_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp26'))
sp26_df$FactorCancers <- factor(sp26_df$Cancer_name, levels = unique(sp26_df$Cancer_name))
sp26 = sp26_df$Attention
# sp27
sp27_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp27'))
sp27_df$FactorCancers <- factor(sp27_df$Cancer_name, levels = unique(sp27_df$Cancer_name))
sp27 = sp27_df$Attention
# sp28
sp28_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp28'))
sp28_df$FactorCancers <- factor(sp28_df$Cancer_name, levels = unique(sp28_df$Cancer_name))
sp28 = sp28_df$Attention
# sp29
sp29_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp29'))
sp29_df$FactorCancers <- factor(sp29_df$Cancer_name, levels = unique(sp29_df$Cancer_name))
sp29 = sp29_df$Attention
# sp30
sp30_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp30'))
sp30_df$FactorCancers <- factor(sp30_df$Cancer_name, levels = unique(sp30_df$Cancer_name))
sp30 = sp30_df$Attention
# sp31
sp31_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp31'))
sp31_df$FactorCancers <- factor(sp31_df$Cancer_name, levels = unique(sp31_df$Cancer_name))
sp31 = sp31_df$Attention
# sp32
sp32_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp32'))
sp32_df$FactorCancers <- factor(sp32_df$Cancer_name, levels = unique(sp32_df$Cancer_name))
sp32 = sp32_df$Attention
# sp33
sp33_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp33'))
sp33_df$FactorCancers <- factor(sp33_df$Cancer_name, levels = unique(sp33_df$Cancer_name))
sp33 = sp33_df$Attention
# sp34
sp34_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp34'))
sp34_df$FactorCancers <- factor(sp34_df$Cancer_name, levels = unique(sp34_df$Cancer_name))
sp34 = sp34_df$Attention
# sp35
sp35_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp35'))
sp35_df$FactorCancers <- factor(sp35_df$Cancer_name, levels = unique(sp35_df$Cancer_name))
sp35 = sp35_df$Attention
# sp36
sp36_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp36'))
sp36_df$FactorCancers <- factor(sp36_df$Cancer_name, levels = unique(sp36_df$Cancer_name))
sp36 = sp36_df$Attention
# sp37
sp37_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp37'))
sp37_df$FactorCancers <- factor(sp37_df$Cancer_name, levels = unique(sp37_df$Cancer_name))
sp37 = sp37_df$Attention
# sp38
sp38_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp38'))
sp38_df$FactorCancers <- factor(sp38_df$Cancer_name, levels = unique(sp38_df$Cancer_name))
sp38 = sp38_df$Attention
# sp39
sp39_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp39'))
sp39_df$FactorCancers <- factor(sp39_df$Cancer_name, levels = unique(sp39_df$Cancer_name))
sp39 = sp39_df$Attention
# sp40
sp40_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp40'))
sp40_df$FactorCancers <- factor(sp40_df$Cancer_name, levels = unique(sp40_df$Cancer_name))
sp40 = sp40_df$Attention
# sp41
sp41_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp41'))
sp41_df$FactorCancers <- factor(sp41_df$Cancer_name, levels = unique(sp41_df$Cancer_name))
sp41 = sp41_df$Attention
# sp42
sp42_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp42'))
sp42_df$FactorCancers <- factor(sp42_df$Cancer_name, levels = unique(sp42_df$Cancer_name))
sp42 = sp42_df$Attention
# sp43
sp43_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp43'))
sp43_df$FactorCancers <- factor(sp43_df$Cancer_name, levels = unique(sp43_df$Cancer_name))
sp43 = sp43_df$Attention
# sp44
sp44_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp44'))
sp44_df$FactorCancers <- factor(sp44_df$Cancer_name, levels = unique(sp44_df$Cancer_name))
sp44 = sp44_df$Attention
# sp45
sp45_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp45'))
sp45_df$FactorCancers <- factor(sp45_df$Cancer_name, levels = unique(sp45_df$Cancer_name))
sp45 = sp45_df$Attention
# sp46
sp46_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp46'))
sp46_df$FactorCancers <- factor(sp46_df$Cancer_name, levels = unique(sp46_df$Cancer_name))
sp46 = sp46_df$Attention
# sp47
sp47_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp47'))
sp47_df$FactorCancers <- factor(sp47_df$Cancer_name, levels = unique(sp47_df$Cancer_name))
sp47 = sp47_df$Attention
# sp48
sp48_df = hop1_edge_filtered_average_join_merge_df %>% filter(SpNotation %in% c('sp48'))
sp48_df$FactorCancers <- factor(sp48_df$Cancer_name, levels = unique(sp48_df$Cancer_name))
sp48 = sp48_df$Attention








tracks = BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions,
                                        sp1, minRadius = 0.2, maxRadius = 0.216, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp2, minRadius = 0.216, maxRadius = 0.232, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp3, minRadius = 0.232, maxRadius = 0.248, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp4, minRadius = 0.248, maxRadius = 0.264, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp5, minRadius = 0.264, maxRadius = 0.28, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp6, minRadius = 0.28, maxRadius = 0.296, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp7, minRadius = 0.296, maxRadius = 0.312, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp8, minRadius = 0.312, maxRadius = 0.328, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp9, minRadius = 0.328, maxRadius = 0.344, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp10, minRadius = 0.344, maxRadius = 0.36, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp11, minRadius = 0.36, maxRadius = 0.376, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp12, minRadius = 0.376, maxRadius = 0.392, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp13, minRadius = 0.392, maxRadius = 0.408, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp14, minRadius = 0.408, maxRadius = 0.424, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp15, minRadius = 0.424, maxRadius = 0.44, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp16, minRadius = 0.44, maxRadius = 0.456, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp17, minRadius = 0.456, maxRadius = 0.472, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp18, minRadius = 0.472, maxRadius = 0.488, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp19, minRadius = 0.488, maxRadius = 0.504, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp20, minRadius = 0.504, maxRadius = 0.52, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp21, minRadius = 0.52, maxRadius = 0.536, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp22, minRadius = 0.536, maxRadius = 0.552, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp23, minRadius = 0.552, maxRadius = 0.568, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp24, minRadius = 0.568, maxRadius = 0.584, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp25, minRadius = 0.584, maxRadius = 0.6, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp26, minRadius = 0.6, maxRadius = 0.616, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp27, minRadius = 0.616, maxRadius = 0.632, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp28, minRadius = 0.632, maxRadius = 0.648, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp29, minRadius = 0.648, maxRadius = 0.664, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp30, minRadius = 0.664, maxRadius = 0.68, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp31, minRadius = 0.68, maxRadius = 0.696, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp32, minRadius = 0.696, maxRadius = 0.712, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp33, minRadius = 0.712, maxRadius = 0.728, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp34, minRadius = 0.728, maxRadius = 0.744, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp35, minRadius = 0.744, maxRadius = 0.76, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp36, minRadius = 0.76, maxRadius = 0.776, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp37, minRadius = 0.776, maxRadius = 0.792, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp38, minRadius = 0.792, maxRadius = 0.808, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp39, minRadius = 0.808, maxRadius = 0.824, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp40, minRadius = 0.824, maxRadius = 0.84, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp41, minRadius = 0.84, maxRadius = 0.856, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp42, minRadius = 0.856, maxRadius = 0.872, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp43, minRadius = 0.872, maxRadius = 0.888, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp44, minRadius = 0.904, maxRadius = 0.92, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp45, minRadius = 0.92, maxRadius = 0.936, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp46, minRadius = 0.936, maxRadius = 0.952, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp47, minRadius = 0.952, maxRadius = 0.968, color = c("#FFAAAA", "white"))
tracks = tracks + BioCircosHeatmapTrack("heatmap1", cancer_box, cancer_box_positions - 1, cancer_box_positions, 
                                        sp48, minRadius = 0.968, maxRadius = 0.984, color = c("#FFAAAA", "white"))


BioCircos(tracks, 
          starts = c(90),
          ends = c(270),
          genome = as.list(cancer_types_counts),
          genomeTicksDisplay = F,
          genomeLabelDy = 0, 
          HEATMAPMouseOverColor = "#F3C73A")







