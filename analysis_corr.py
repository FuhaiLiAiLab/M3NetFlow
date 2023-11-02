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

class RebuildLoss():
    def __init__(self):
        pass

    def rebuild_loss_pearson(self, path, epoch_num):
        test_epoch_loss_list = []
        train_epoch_loss_list = []
        test_epoch_pearson_list = []
        train_epoch_pearson_list = []
        min_test_loss = 1000
        min_train_loss = 1000
        max_test_corr = 0
        max_train_corr = 0
        max_test_id = 0
        for i in range(1, epoch_num + 1):
            # TEST LOSS
            test_df = pd.read_csv(path + '/TestPred' + str(i) + '.txt', delimiter=',')
            test_score_list = list(test_df['Score'])
            test_pred_list = list(test_df['Pred Score'])
            test_epoch_loss = mean_squared_error(test_score_list, test_pred_list)
            test_epoch_loss_list.append(test_epoch_loss)
            test_epoch_pearson = test_df.corr(method = 'pearson')
            test_epoch_corr = test_epoch_pearson['Pred Score'][0]
            test_epoch_pearson_list.append(test_epoch_corr)
            # TRAIN LOSS
            train_df = pd.read_csv(path + '/TrainingPred_' + str(i) + '.txt', delimiter=',')
            train_score_list = list(train_df['Score'])
            train_pred_list = list(train_df['Pred Score'])
            train_epoch_loss = mean_squared_error(train_score_list, train_pred_list)
            train_epoch_loss_list.append(train_epoch_loss)
            train_epoch_pearson = train_df.corr(method = 'pearson')
            train_epoch_corr = train_epoch_pearson['Pred Score'][0]
            train_epoch_pearson_list.append(train_epoch_corr)
            if test_epoch_loss < min_test_loss:
                min_test_loss = test_epoch_loss
                min_train_loss = train_epoch_loss
            if test_epoch_corr > max_test_corr:
                max_test_corr = test_epoch_corr
                max_train_corr = train_epoch_corr
                max_test_id = i
        best_train_df = pd.read_csv(path + '/TrainingPred_' + str(max_test_id) + '.txt', delimiter=',')
        best_train_df.to_csv(path + '/BestTrainingPred.txt', index=False, header=True)
        best_test_df = pd.read_csv(path + '/TestPred' + str(max_test_id) + '.txt', delimiter=',')
        best_test_df.to_csv(path + '/BestTestPred.txt', index=False, header=True)
        # import pdb; pdb.set_trace()
        print('-------------BEST MODEL ID:' + str(max_test_id) + '-------------')
        print('-------------BEST MODEL ID:' + str(max_test_id) + '-------------')
        print('-------------BEST MODEL ID:' + str(max_test_id) + '-------------')
        print('BEST MODEL TRAIN LOSS: ', min_train_loss)
        print('BEST MODEL PEARSON CORR: ', max_train_corr)
        print('BEST MODEL TEST LOSS: ', min_test_loss)
        print('BEST MODEL PEARSON CORR: ', max_test_corr)
        epoch_pearson_array = np.array(test_epoch_pearson_list)
        epoch_loss_array = np.array(test_epoch_loss_list)
        np.save(path + '/pearson.npy', epoch_pearson_array)
        np.save(path + '/loss.npy', epoch_loss_array)
        return max_test_id


class AnalyseCorr():
    def __init__(self):
        pass

    def pred_result(self, fold_n, epoch_name, dataset, modelname):
        plot_path = './' + dataset + '/plot' + '/' + modelname
        if os.path.exists(plot_path) == False:
            os.mkdir(plot_path)
        ### TRAIN PRED JOINTPLOT
        train_pred_df = pd.read_csv('./' + dataset + '/result/' + modelname + '/' + epoch_name + '/BestTrainingPred.txt')
        sns.set_style('whitegrid')
        sns.jointplot(data=train_pred_df, x='Score', y='Pred Score', kind='reg')
        train_pearson = train_pred_df.corr(method='pearson')['Pred Score'][0]
        train_score_list = list(train_pred_df['Score'])
        train_pred_list = list(train_pred_df['Pred Score'])
        train_loss = mean_squared_error(train_score_list, train_pred_list)
        plt.legend(['Training Pearson =' + str(train_pearson)])
        plt.savefig(plot_path + '/trainpred_corr_' + str(fold_n) + '.png', dpi=300)
        ### TEST PRED JOINTPLOT
        test_pred_df = pd.read_csv('./' + dataset + '/result/' + modelname + '/' + epoch_name + '/BestTestPred.txt')
        comb_testpred_df = pd.read_csv('./' + dataset + '/filtered_data/split_input_' + str(fold_n) + '.csv')
        comb_testpred_df['Pred Score'] = list(test_pred_df['Pred Score'])
        comb_testpred_df.to_csv('./' + dataset + '/result/' + modelname + '/' + epoch_name + '/combine_testpred.csv', index=False, header=True)
        sns.set_style('whitegrid')
        sns.jointplot(data=comb_testpred_df, x='Score', y='Pred Score', kind='reg')
        test_pearson = test_pred_df.corr(method='pearson')['Pred Score'][0]
        test_score_list = list(test_pred_df['Score'])
        test_pred_list = list(test_pred_df['Pred Score'])
        test_loss = mean_squared_error(test_score_list, test_pred_list)
        plt.legend(['Test Pearson =' + str(test_pearson)])
        plt.savefig(plot_path + '/testpred_corr_' + str(fold_n) + '.png', dpi=300)
        print('--- TRAIN ---')
        print('BEST MODEL TRAIN LOSS: ', train_loss)
        print('BEST MODEL TRAIN PEARSON CORR: ', train_pearson)
        print('--- TEST ---')
        print('BEST MODEL TEST LOSS: ', test_loss)
        print('BEST MODEL TEST PEARSON CORR: ', test_pearson)
        ### HISTOGRAM
        hist = test_pred_df.hist(column=['Score', 'Pred Score'], bins=20)
        plt.savefig(plot_path + '/testpred_hist_' + str(fold_n) + '.png', dpi=300)
        ### BOX PLOT
        testpred_df = comb_testpred_df[['Cell Line Name', 'Pred Score']]
        testpred_df['Type'] = ['Prediction Score']*testpred_df.shape[0]
        testpred_df = testpred_df.rename(columns={'Pred Score': 'Drug Score'})
        test_df = comb_testpred_df[['Cell Line Name', 'Score']]
        test_df['Type'] = ['Input Score']*test_df.shape[0]
        test_df = test_df.rename(columns={'Score': 'Drug Score'})
        comb_score_df = pd.concat([testpred_df, test_df])
        comb_score_df = comb_score_df.rename(columns={'Cell Line Name': 'cell_line_name'})
        a4_dims = (20, 15)
        fig, ax = plt.subplots(figsize=a4_dims)
        sns.set_context('paper')
        # import pdb; pdb.set_trace()
        cell_line_list = sorted(list(set(comb_score_df['cell_line_name'])))
        sns.boxplot(ax=ax, x='cell_line_name', y='Drug Score', hue='Type', data=comb_score_df, order=cell_line_list)
        plt.xticks(rotation = 90, ha = 'right')
        plt.savefig(plot_path + '/testpred_compare_cell_line_boxplot_' + str(fold_n) + '.png', dpi=600)
        plt.close('all')
        # plt.show()
        return train_pearson, test_pearson
    
    def pred_all_result(self, num_fold, epoch_num, dataset, modelname, train_mean=True):
        plot_path = './' + dataset + '/plot' + '/' + modelname
        if os.path.exists(plot_path) == False:
            os.mkdir(plot_path)
        ### SCATTER PLOT OF ALL FOLD ON TEST / TRAIN
        # TEST
        test_pred_df_list = []
        for fold_n in range(1, num_fold + 1):
            if fold_n == 1:
                epoch_name = 'epoch_' + str(epoch_num)
            else:
                epoch_name = 'epoch_' + str(epoch_num) + '_' + str(fold_n-1)
            test_pred_df = pd.read_csv('./' + dataset + '/result/' + modelname + '/' + epoch_name + '/combine_testpred.csv')
            test_pred_df_list.append(test_pred_df)
        comb_testpred_df = pd.concat(test_pred_df_list)
        comb_testpred_df = comb_testpred_df.rename(columns={'Cell Line Name': 'cell_line_name'}).reset_index(drop=True)
        print(comb_testpred_df)
        sns.set_style('whitegrid')
        sns.jointplot(data=comb_testpred_df, x='Score', y='Pred Score', kind='reg')
        comb_test_pearson = comb_testpred_df.corr(method='pearson')['Pred Score'][0]
        comb_test_score_list = list(comb_testpred_df['Score'])
        comb_test_pred_list = list(comb_testpred_df['Pred Score'])
        comb_test_loss = mean_squared_error(comb_test_score_list, comb_test_pred_list)
        print('COMBINED MODEL TEST LOSS: ', comb_test_loss)
        print('COMBINED MODEL TEST PEARSON CORR: ', comb_test_pearson)
        plt.legend(['Test Pearson =' + str(comb_test_pearson)])
        plt.savefig(plot_path + '/comb_test_corr.png', dpi=300)
        # TRAIN
        train_pred_df_list = []
        for fold_n in range(1, num_fold + 1):
            if fold_n == 1:
                epoch_name = 'epoch_' + str(epoch_num)
            else:
                epoch_name = 'epoch_' + str(epoch_num) + '_' + str(fold_n-1)
            train_predscore_df = pd.read_csv('./' + dataset + '/result/' + modelname + '/' + epoch_name + '/BestTrainingPred.txt')
            train_pred_df = pd.read_csv('./' + dataset + '/filtered_data/split_train_input_' + str(fold_n) + '.csv')
            train_pred_df['Pred Score'] = list(train_predscore_df['Pred Score'])
            train_pred_df.to_csv('./' + dataset + '/result/' + modelname + '/' + epoch_name + '/combine_trainpred.csv', index=False, header=True)
            train_pred_df_list.append(train_pred_df)
        comb_trainpred_df = pd.concat(train_pred_df_list)
        comb_trainpred_df = comb_trainpred_df.rename(columns={'Cell Line Name': 'cell_line_name'}).reset_index(drop=True)
        print(comb_trainpred_df)
        if train_mean == True:
            comb_trainpred_df = comb_trainpred_df.groupby(['Drug A', 'Drug B', 'cell_line_name']).mean().reset_index()
        print(comb_trainpred_df)
        sns.set_style('whitegrid')
        sns.jointplot(data=comb_trainpred_df, x='Score', y='Pred Score', kind='reg')
        comb_train_pearson = comb_trainpred_df.corr(method='pearson')['Pred Score'][0]
        comb_train_score_list = list(comb_trainpred_df['Score'])
        comb_train_pred_list = list(comb_trainpred_df['Pred Score'])
        comb_train_loss = mean_squared_error(comb_train_score_list, comb_train_pred_list)
        print('COMBINED MODEL TRAIN LOSS: ', comb_train_loss)
        print('COMBINED MODEL TRAIN PEARSON CORR: ', comb_train_pearson)
        plt.legend(['Train Pearson =' + str(comb_train_pearson)])
        plt.savefig(plot_path + '/comb_train_corr.png', dpi=300)
        # plt.show()
        ### BOX PLOT
        testpred_df = comb_testpred_df[['cell_line_name', 'Pred Score']]
        testpred_df['Type'] = ['Prediction Score']*testpred_df.shape[0]
        testpred_df = testpred_df.rename(columns={'Pred Score': 'Drug Score'})
        test_df = comb_testpred_df[['cell_line_name', 'Score']]
        test_df['Type'] = ['Input Score']*test_df.shape[0]
        test_df = test_df.rename(columns={'Score': 'Drug Score'})
        comb_score_df = pd.concat([testpred_df, test_df])
        a4_dims = (15, 15)
        fig, ax = plt.subplots(figsize=a4_dims)
        sns.set_context('paper', font_scale=1.5)
        sns.set_palette("Set2")  
        # cell_line_list = sorted(list(set(comb_score_df['cell_line_name'])))
        final_dl_input_df = pd.read_csv('./' + dataset + '/filtered_data/final_dl_input.csv')
        cell_line_list = final_dl_input_df['Cell Line Name'].value_counts().index
        sns.boxplot(ax=ax, x='cell_line_name', y='Drug Score', hue='Type', data=comb_score_df, order=cell_line_list, width=0.6)
        ax.set_xticklabels(cell_line_list, fontsize=13)  
        plt.xticks(rotation = 90, ha = 'right')
        ax.set_xlabel('Cell Line Names', fontsize=16)  # Adjust the fontsize value as needed
        ax.set_ylabel('Drug Score', fontsize=16)  # Adjust the fontsize value as needed
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.9)
        plt.savefig(plot_path + '/testpred_compare_cell_line_boxplot_all.png', dpi=600)
        plt.close('all')
        # plt.show()
        # return train_pearson, test_pearson

    def comparison(self, dataset, gcn_decoder_test_list, gat_decoder_test_list, m3net_decoder_test_list):
        colors = sns.color_palette("Set2", 3)
        labels = ['1st Fold', '2nd Fold', '3rd Fold', '4th Fold', '5th Fold', 'Average']
        x = np.arange(len(labels))
        width = 0.25 
        print(gcn_decoder_test_list)
        print(gat_decoder_test_list)
        print(m3net_decoder_test_list)
        sns.set_style(style=None)
        gcn = plt.bar(x - 1*width, gcn_decoder_test_list, width, label='GCN Decoder', color=colors[0])
        gat = plt.bar(x, gat_decoder_test_list, width, label='GAT Decoder', color=colors[1])
        m3net = plt.bar(x + 1*width, m3net_decoder_test_list, width, label='M3NetFlow', color=colors[2])
        plt.ylabel('Pearson Correlation')
        # plt.title('Pearson Correlation Comparison For 3 GNN Models')
        plt.ylim(0.0, 0.8)
        plt.xticks(x, labels=labels)
        plt.legend()
        plt.savefig('./' + dataset + '/result/comparisons.png', dpi=600)
        # plt.show()


def model_result(dataset, modelname, epoch_num, rebuild=True):
    model_test_result_list = []
    for fold_n in np.arange(1, 6):
        fold_num = str(fold_n) + 'th'
        if fold_n == 1:
            epoch_name = 'epoch_' + str(epoch_num)
        else:
            epoch_name = 'epoch_' + str(epoch_num) + '_' + str(fold_n-1)
        if os.path.exists('./' + dataset + '/plot') == False:
            os.mkdir('./' + dataset + '/plot')
        # REBUILD BEST ID
        if rebuild == True:
            path = './' + dataset + '/result/' + modelname + '/' + epoch_name
            max_test_id = RebuildLoss().rebuild_loss_pearson(path, epoch_num)
        train_path = './' + dataset + '/result/' + modelname + '/' + epoch_name + '/BestTrainingPred.txt'
        test_path = './' + dataset + '/result/' + modelname + '/' + epoch_name + '/BestTestPred.txt'
        train_pearson, test_pearson = AnalyseCorr().pred_result(fold_n=fold_n, epoch_name=epoch_name, dataset=dataset, modelname=modelname)
        model_test_result_list.append(test_pearson)
    average_test_result = sum(model_test_result_list) / len(model_test_result_list)
    model_test_result_list.append(average_test_result)
    print(model_test_result_list)
    return model_test_result_list

def train_test_split(num_fold=5, dataset='datainfo-nci'):
    for fold_n in range(1, num_fold + 1):
        filtered_data_path = './' + dataset + '/filtered_data'
        train_df_list = []
        for i in range(1, num_fold + 1):
            print(i)
            if i == fold_n:
                print('--- LOADING ' + str(i) + '-TH SPLIT TEST DATA ---')
                test_df = pd.read_csv(filtered_data_path + '/split_input_' + str(i) + '.csv')
            else:
                print('--- LOADING ' + str(i) + '-TH SPLIT TRAINING DATA ---')
                train_df = pd.read_csv(filtered_data_path + '/split_input_' + str(i) + '.csv')
                train_df_list.append(train_df)
        concat_train_df = pd.concat(train_df_list)
        test_df.to_csv(filtered_data_path + '/split_test_input_' + str(fold_n) + '.csv', index=False, header=True)
        concat_train_df.to_csv(filtered_data_path + '/split_train_input_' + str(fold_n) + '.csv', index=False, header=True)
        print(test_df.shape)
        print(concat_train_df.shape)

def cell_line_cancer_percentage(dataset):
    final_dl_input_df = pd.read_csv('./' + dataset + '/filtered_data/final_dl_input.csv')
    cell_line_cancer_name_map_dict_df = pd.read_csv('./' + dataset + '/filtered_data/cell_line_cancer_name_map_dict.csv')
    final_dl_input_cancer_df = pd.merge(final_dl_input_df, cell_line_cancer_name_map_dict_df, left_on='Cell Line Name', right_on='Cell_Line_Name')
    print(final_dl_input_cancer_df)

    # MAP EACH CELL LINE TO ITS CANCER TYPE
    cell_line_cancer_dict = dict(map(lambda i, j : (i, j) , cell_line_cancer_name_map_dict_df.Cell_Line_Name, cell_line_cancer_name_map_dict_df.Cancer_name))
    # ASSIGN A COLOR FOR EACH CANCER TYPE
    # color_list = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3', '#bcbd22']
    color_list = plt.cm.tab20(np.linspace(0, 1, len(list(set(list(cell_line_cancer_dict.values()))))))
    cancer_to_color_dict = dict(map(lambda i, j : (i, j) , list(set(list(cell_line_cancer_dict.values()))), color_list))

    # HORIZONTAL BAR PLOT
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.barh(final_dl_input_df['Cell Line Name'].value_counts().index, 
            final_dl_input_df['Cell Line Name'].value_counts().values, 
            color=[cancer_to_color_dict[cell_line_cancer_dict[cell]] for cell in final_dl_input_df['Cell Line Name'].value_counts().index])
    ax1.set_xlabel('Count')
    ax1.set_ylabel('Cell Line')
    ax1.set_title('Count of Cell Lines')
    plt.tight_layout()
    ax1.invert_yaxis()

    # PIE CHART FOR CANCER TYPES
    cancer_type_percentages = final_dl_input_cancer_df['Cancer_name'].value_counts(normalize=True) * 100
    colors = [cancer_to_color_dict[cancer] for cancer in cancer_type_percentages.index]
    plt.figure(figsize=(15, 9))
    cancer_type_percentages.plot(kind='pie', ax=ax2, autopct='%1.1f%%', startangle=140, colors=colors, textprops={'fontsize': 14})
    ax2.set_title('Percentage of Cancer Types')
    ax2.set_ylabel('')
    plt.show()

def cell_line_cancer_percentage_split(dataset):
    final_dl_input_df = pd.read_csv('./' + dataset + '/filtered_data/final_dl_input.csv')
    cell_line_cancer_name_map_dict_df = pd.read_csv('./' + dataset + '/filtered_data/cell_line_cancer_name_map_dict.csv')
    final_dl_input_cancer_df = pd.merge(final_dl_input_df, cell_line_cancer_name_map_dict_df, left_on='Cell Line Name', right_on='Cell_Line_Name')
    print(final_dl_input_cancer_df)

    # MAP EACH CELL LINE TO ITS CANCER TYPE
    cell_line_cancer_dict = dict(map(lambda i, j : (i, j) , cell_line_cancer_name_map_dict_df.Cell_Line_Name, cell_line_cancer_name_map_dict_df.Cancer_name))
    # ASSIGN A COLOR FOR EACH CANCER TYPE
    color_list = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3', '#bcbd22']
    # color_list = plt.cm.tab10(np.linspace(0, 1, len(list(set(list(cell_line_cancer_dict.values()))))))
    cancer_to_color_dict = dict(map(lambda i, j : (i, j) , list(set(list(cell_line_cancer_dict.values()))), color_list))

    # HORIZONTAL BAR PLOT
    fig1, ax1 = plt.subplots(figsize=(12, 15))
    colors_bar = [cancer_to_color_dict[cell_line_cancer_dict[cell]] for cell in final_dl_input_df['Cell Line Name'].value_counts().index]
    ax1.barh(final_dl_input_df['Cell Line Name'].value_counts().index, 
            final_dl_input_df['Cell Line Name'].value_counts().values, 
            color=colors_bar)
    ax1.set_xlabel('Count')
    ax1.set_ylabel('Cell Line')
    ax1.set_title('Count of Cell Lines')
    ax1.invert_yaxis()
    plt.tight_layout()
    plt.savefig('./' + dataset + '/result/cell_line_barplot.png', dpi=600)
    # plt.show()  # This will display the first plot


    # PIE CHART FOR CANCER TYPES
    fig2, ax2 = plt.subplots(figsize=(15, 9))
    cancer_type_percentages = final_dl_input_cancer_df['Cancer_name'].value_counts(normalize=True) * 100
    colors_pie = [cancer_to_color_dict[cancer] for cancer in cancer_type_percentages.index]
    cancer_type_percentages.plot(kind='pie', ax=ax2, autopct='%1.1f%%', startangle=140, colors=colors_pie, textprops={'fontsize': 18})
    ax2.set_title('Percentage of Cancer Types')
    ax2.set_ylabel('')
    plt.savefig('./' + dataset + '/result/cancer_cell_line_pieplot.png', dpi=600)
    # plt.show()  # This will display the second plot


# Custom function to rotate the labels
def func(pct, allvalues): 
    absolute = int(pct / 100.*np.sum(allvalues)) 
    return "{:.1f}%\n".format(pct, absolute)


if __name__ == "__main__":
    ### DATASET SELECTION
    dataset = 'datainfo-nci'
    # dataset = 'datainfo-oneil'

    # train_test_split(num_fold=5, dataset=dataset)

    rebuild = False

    ### MODEL SELECTION
    # gcn_decoder_test_list = model_result(dataset=dataset, modelname='gcn', epoch_num=180, rebuild=rebuild) 
    # gat_decoder_test_list = model_result(dataset=dataset, modelname='gat', epoch_num=100, rebuild=rebuild) 
    # m3net_decoder_test_list = model_result(dataset=dataset, modelname='tsgnn', epoch_num=100, rebuild=rebuild) 

    # ### NCI-ALMANAC
    # gcn_decoder_test_list = [0.5064605657342812, 0.5245598748183634, 0.46695701896538616, 0.5214272962811511, 0.577206595098155, 0.5193222701794674]
    # gat_decoder_test_list = [0.4917665201363847, 0.48933140810934184, 0.5279742511264676, 0.4924140826253474, 0.45652443502586376, 0.4916021394046811]
    # m3net_decoder_test_list = [0.6009660675416006, 0.6010748931833301, 0.617941461925096, 0.6009383601682954, 0.6151593354988303, 0.6072160236634304]

    # ### O'NEIL
    # gcn_decoder_test_list = [0.43974780918316303, 0.32089903988461155, 0.48944945583649246, 0.46613296163029383, 0.5075061443622683, 0.4447470821793658]
    # gat_decoder_test_list = [0.5213588523120589, 0.5890929146467674, 0.5305813326489733, 0.6111279005550803, 0.6009412335511605, 0.570620446742808]
    # m3net_decoder_test_list = [0.629468193556913, 0.6383564434819521, 0.616114257622284, 0.6906363694186656, 0.6432778487211956, 0.643570622560202]

    # AnalyseCorr().comparison(dataset, gcn_decoder_test_list, gat_decoder_test_list, m3net_decoder_test_list)


    # AnalyseCorr().pred_result(fold_n=1, epoch_name='epoch_100', dataset=dataset, modelname='tsgnn')
    # AnalyseCorr().pred_all_result(num_fold=5, epoch_num=100, dataset=dataset, modelname='tsgnn', train_mean=False)

    # cell_line_cancer_percentage(dataset=dataset)
    cell_line_cancer_percentage_split(dataset=dataset)