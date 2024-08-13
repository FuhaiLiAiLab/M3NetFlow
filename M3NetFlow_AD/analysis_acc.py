import os
import pdb
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.pyplot import figure
from sklearn.metrics import mean_squared_error
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class AnalyseCorr():
    def __init__(self):
        pass

    def dataset_avg_comparison(self, gcn_decoder_avg_list, gat_decoder_avg_list, m3net_decoder_avg_list,
                        unimp_decoder_avg_list, mixhop_decoder_avg_list, pna_decoder_avg_list, gin_decoder_avg_list):
        colors = sns.color_palette("Set2", 7)
        labels = ['NCI ALMANAC', 'O\'Neil', 'ROSMAP']
        x = np.arange(len(labels))
        width = 0.1
        print(gcn_decoder_avg_list)
        print(gat_decoder_avg_list)
        print(unimp_decoder_avg_list)
        print(mixhop_decoder_avg_list)
        print(pna_decoder_avg_list)
        print(gin_decoder_avg_list)
        print(m3net_decoder_avg_list)
        sns.set_style(style=None)
        gcn = plt.bar(x - 3*width, gcn_decoder_avg_list, width, label='GCN', color=colors[0])
        gat = plt.bar(x - 2*width, gat_decoder_avg_list, width, label='GAT', color=colors[1])
        unimp = plt.bar(x - 1*width, unimp_decoder_avg_list, width, label='UniMP', color=colors[2])
        mixhop = plt.bar(x, mixhop_decoder_avg_list, width, label='MixHop', color=colors[3])
        pna = plt.bar(x + 1*width, pna_decoder_avg_list, width, label='PNA', color=colors[4])
        gin = plt.bar(x + 2*width, gin_decoder_avg_list, width, label='GIN', color=colors[5])
        m3net = plt.bar(x + 3*width, m3net_decoder_avg_list, width, label='M3NetFlow', color=colors[6])
        plt.ylabel('Pearson Correlation / Accuracy')
        # plt.title('Pearson Correlation Comparison For 3 GNN Models')
        plt.ylim(0.0, 0.9)
        plt.xticks(x, labels=labels)
        plt.legend(loc='upper right', ncol=2)
        plt.savefig('./ROSMAP-result/dataset_avg_comparisons.png', dpi=600)
        # plt.show()


if __name__ == "__main__":
    ### DATASET SELECTION
    dataset = 'ROSMAP'
    rebuild = False
    
    # ### DATASET SCORES
    gcn_decoder_avg_list = [0.5193222701794674, 0.4447470821793658, 0.5943]
    gat_decoder_avg_list = [0.4916021394046811, 0.570620446742808, 0.628]
    unimp_decoder_avg_list = [0.49022426040184114, 0.558423216427899, 0.6183]
    mixhop_decoder_avg_list = [0.5777981131221722, 0.2715167302410376, 0.572]
    pna_decoder_avg_list = [0.556311605369027, 0.6220330697323424, 0.5783]
    gin_decoder_avg_list = [0.537620272333274, 0.331249026017392, 0.4983]
    m3net_decoder_avg_list = [0.6072160236634304, 0.643570622560202, 0.6583]

    AnalyseCorr().dataset_avg_comparison( 
                                    gcn_decoder_avg_list, 
                                    gat_decoder_avg_list, 
                                    m3net_decoder_avg_list, 
                                    unimp_decoder_avg_list, 
                                    mixhop_decoder_avg_list, 
                                    pna_decoder_avg_list, 
                                    gin_decoder_avg_list)
