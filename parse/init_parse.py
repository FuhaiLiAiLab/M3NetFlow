import os
import re
import numpy as np
import pandas as pd

from numpy import savetxt
from sklearn.model_selection import train_test_split

def minMax(x): return pd.Series(index=['min','max'],data=[x.min(),x.max()])

class ReadFile():
    def __init__(self):
        pass

    def combo_input(self, dataset):
        ### INTIALIZE [NCI60 DrugScreen Data]
        print('----- READING NCI60 DRUG SCREEN RAW DATA -----')
        if os.path.exists('../' + dataset + '/init_data') == False:
            os.mkdir('../' + dataset + '/init_data')
        dl_input_df = pd.read_csv('../' + dataset + '/raw_data/NCI60/DeepLearningInput.csv')
        dl_input_df = dl_input_df.groupby(['Drug A', 'Drug B', 'Cell Line Name']).agg({'Score':'mean'}).reset_index()
        # REMOVE SINGLE DRUG SCREEN DATA [Actual Fact Shows No Single Drug]
        dl_input_deletion_list = []
        dl_input_df = dl_input_df.fillna('missing')
        for row in dl_input_df.itertuples():
            if row[1] == 'missing' or row[2] == 'missing':
                dl_input_deletion_list.append(row[0])
        dl_input_df = dl_input_df.drop(dl_input_df.index[dl_input_deletion_list]).reset_index(drop=True)
        dl_input_df.to_csv('../' + dataset + '/init_data/almanac_dl_input.csv', index=False, header=True)
        ### PROFILE [Number of Drugs / Number of Cell Lines]
        drug_list = list(set(list(dl_input_df['Drug A']) + list(dl_input_df['Drug B'])))
        cell_line_list = list(set(list(dl_input_df['Cell Line Name'])))
        print('----- NUMBER OF DRUGS IN NCI ALMANAC: ' + str(len(drug_list)) + ' -----')
        print('----- NUMBER OF CELL LINES IN NCI ALMANAC: ' + str(len(cell_line_list)) + ' -----')
        print(dl_input_df.shape)

    def parse_drugcomb_fi(self, dataset):
        ### INTIALIZE [O'NEIL DrugScreen Data]
        print('----- READING DrugComb_fi DRUG SCREEN RAW DATA -----')
        if os.path.exists('../' + dataset + '/init_data') == False:
            os.mkdir('../' + dataset + '/init_data')
        drugcomb_fi_df = pd.read_csv('../' + dataset + '/raw_data/DrugComb_fi/summary_v_1_5.csv')
        ### SELECT ONLY [ONEIL] DATASET
        oneil_df = drugcomb_fi_df.loc[drugcomb_fi_df['study_name'] == 'ONEIL']
        oneil_comb_df = oneil_df.loc[oneil_df['drug_col'].notnull()]
        oneil_comb_df['synergy_loewe'] = oneil_comb_df['synergy_loewe'].astype(float)
        # CHECK DRUG COMBINATION SCORE IN DIFFERENT MATRICS
        oneil_comb_num_df = oneil_comb_df[['ic50_row', 'ic50_col', 'ri_row', 'ri_col', 'css_row', 'css_col', 'css_ri', 'S_sum', 'S_mean','S_max', 'synergy_zip', 'synergy_loewe', 'synergy_hsa', 'synergy_bliss']]
        oneil_comb_num_df.apply(minMax)
        # AVERAGE THE SCORE IN [synergy_loewe]
        oneil_comb_syn_df = oneil_comb_df[['block_id', 'drug_row', 'drug_col', 'cell_line_name', 'synergy_loewe', 'study_name']]
        oneil_comb_syn_df = oneil_comb_syn_df.sort_values(by = ['block_id'])
        oneil_comb_syn_new_df = oneil_comb_syn_df.groupby(['drug_row', 'drug_col', 'cell_line_name']).agg({'synergy_loewe':'mean'}).reset_index()
        oneil_comb_syn_new_df.apply(minMax)
        # ax = oneil_comb_syn_new_df['synergy_loewe'].plot.kde()
        oneil_comb_syn_new_df = oneil_comb_syn_new_df.rename(columns={'drug_row': 'Drug A', 'drug_col': 'Drug B', 
                                                'cell_line_name': 'Cell Line Name', 'synergy_loewe': 'Score'})
        ### DROP ['UWB1289+BRCA1'] CELL LINE, REDUCING NUMBER OF ROWS FROM [22737] TO [22154] 
        oneil_comb_syn_new_df.drop(oneil_comb_syn_new_df[oneil_comb_syn_new_df['Cell Line Name'] == 'UWB1289+BRCA1'].index, inplace = True)
        oneil_comb_syn_new_df.drop(oneil_comb_syn_new_df[oneil_comb_syn_new_df['Cell Line Name'] == 'MSTO'].index, inplace = True)
        ### PROFILE [Number of Drugs / Number of Cell Lines]
        drug_list = list(set(list(oneil_comb_syn_new_df['Drug A']) + list(oneil_comb_syn_new_df['Drug B'])))
        cell_line_list = list(set(list(oneil_comb_syn_new_df['Cell Line Name'])))
        print('----- NUMBER OF DRUGS IN O\'NEIL DATASET: ' + str(len(drug_list)) + ' -----')
        print('----- NUMBER OF CELL LINES O\'NEIL DATASET: ' + str(len(cell_line_list)) + ' -----')
        print(oneil_comb_syn_new_df.shape)
        oneil_comb_syn_new_df.to_csv('../' + dataset + '/init_data/oneil_dl_input.csv', index=False, header=True)
    
    def gdsc_rnaseq(self, dataset):
        ### INTIALIZE [GDSC RNA Sequence Data]
        print('----- READING GDSC RNA Sequence RAW DATA -----')
        rna_df = pd.read_csv('../' + dataset + '/raw_data/GDSC/rnaseq_20191101/rnaseq_fpkm_20191101.csv', low_memory=False)
        rna_df = rna_df.fillna(0.0)
        rna_df.to_csv('../' + dataset + '/init_data/gdsc_rnaseq.csv', index=False, header=True)
        print(rna_df.shape)
        # AFTER THIS NEED SOME MANUAL OPERATIONS TO CHANGE COLUMNS AND ROWS NAMES

    def gdsc_cnv(self, dataset):
        cnv_df = pd.read_csv('../' + dataset + '/raw_data/GDSC/cnv_20191101/cnv_gistic_20191101.csv', low_memory=False)
        cnv_df = cnv_df.fillna(0.0)
        cnv_df.to_csv('../' + dataset + '/init_data/gdsc_cnv.csv', index = False, header = True)
        print(cnv_df.shape)
        # AFTER THIS NEED SOME MANUAL OPERATIONS TO CHANGE COLUMNS AND ROWS NAMES

    def ccle_meth(self, dataset):
        ccle_meth1_df = pd.read_table('../' + dataset + '/raw_data/CCLE/meth/CCLE_RRBS_TSS1kb_20181022.txt', delimiter='\t')
        # ccle_meth1_df = ccle_meth1_df.replace('    NaN', np.nan)
        # ccle_meth1_df = ccle_meth1_df.replace('     NA', np.nan)
        ccle_meth1_df = ccle_meth1_df.replace('    NaN', 0.0)
        ccle_meth1_df = ccle_meth1_df.replace('     NA', 0.0)
        ccle_meth1_df.drop(ccle_meth1_df.tail(1).index, inplace=True)
        print(ccle_meth1_df)
        # REPLACE ALL [locus_id] WITH GENE NAMES
        ccle_meth1_gene_dict = {}
        for row in ccle_meth1_df.itertuples():
            gene = row[1].split('_')[0]
            ccle_meth1_gene_dict[row[1]] = gene
        ccle_meth1_df = ccle_meth1_df.replace({'locus_id': ccle_meth1_gene_dict})
        # REMOVE CERTAIN [CpG_sites_hg19, avg_coverage] COLUMNS
        ccle_meth1_df = ccle_meth1_df.drop(columns=['CpG_sites_hg19', 'avg_coverage'])
        ccle_meth1_df = ccle_meth1_df.sort_values(by=['locus_id']).reset_index(drop=True)
        ccle_meth_column_name_list = list(ccle_meth1_df.columns)[1:]
        ccle_meth1_df = ccle_meth1_df.astype(str)
        # FETCH THE MAXIMUM VALUE WITH REPEATED GENE NAMES
        duplicateRows = ccle_meth1_df[ccle_meth1_df.duplicated(['locus_id'])]
        duplicated_gene_list = sorted(list(set(duplicateRows['locus_id'])))
        renew_max_ccle_meth1_df = ccle_meth1_df.copy()
        renew_min_ccle_meth1_df = ccle_meth1_df.copy()
        count = 0
        for gene_name in duplicated_gene_list:
            tmp_gene_df = ccle_meth1_df[ccle_meth1_df['locus_id'] == gene_name]
            tmp_gene_meth_max_list = list(tmp_gene_df.max())
            tmp_gene_meth_min_list = list(tmp_gene_df.min())
            # import pdb; pdb.set_trace()
            count += 1
            print('----- GENERATING MAXIMUM/MINIMUM METHYLATION VALUE FOR GENE: ' + gene_name + ' IN ' + str(count) + ' -----')
            # DROP [gene_name]
            renew_max_ccle_meth1_df = renew_max_ccle_meth1_df.drop(renew_max_ccle_meth1_df[(renew_max_ccle_meth1_df.locus_id == gene_name)].index).reset_index(drop=True)
            renew_min_ccle_meth1_df = renew_min_ccle_meth1_df.drop(renew_min_ccle_meth1_df[(renew_min_ccle_meth1_df.locus_id == gene_name)].index).reset_index(drop=True)
            # ADD MAX OR MIN VALUE ACROSS
            renew_max_ccle_meth1_df.loc[len(renew_max_ccle_meth1_df)] = tmp_gene_meth_max_list
            renew_min_ccle_meth1_df.loc[len(renew_min_ccle_meth1_df)] = tmp_gene_meth_min_list
        renew_max_ccle_meth1_df = renew_max_ccle_meth1_df.sort_values(by=['locus_id'])
        renew_min_ccle_meth1_df = renew_min_ccle_meth1_df.sort_values(by=['locus_id'])
        print(renew_max_ccle_meth1_df)
        print(renew_min_ccle_meth1_df)
        # # REPLACE ALL WITH FIRST NAME IN CELL LINES
        # ccle_meth1_df = pd.read_table('./' + dataset + '/mid_data/ccle_methylation.txt', delimiter = ',')
        ccle_meth1_cell_line_dict = {}
        ccle_meth_oricell_line = list(ccle_meth1_df.columns)[1:]
        for oricell_line in ccle_meth_oricell_line:
            cell_line = oricell_line.split('_')[0]
            ccle_meth1_cell_line_dict[oricell_line] = cell_line
        renew_max_ccle_meth1_df = renew_max_ccle_meth1_df.rename(columns=ccle_meth1_cell_line_dict)
        renew_min_ccle_meth1_df = renew_min_ccle_meth1_df.rename(columns=ccle_meth1_cell_line_dict)
        # FINALLY [17180 GENES, 843 CELL LINES]
        renew_max_ccle_meth1_df.to_csv('../' + dataset + '/init_data/ccle_methylation_max.csv', index=False, header=True)
        renew_min_ccle_meth1_df.to_csv('../' + dataset + '/init_data/ccle_methylation_min.csv', index=False, header=True)

    def ccle_mutation(self, dataset):
        ccle_mutation_df = pd.read_csv('../' + dataset + '/raw_data/CCLE/mutation/BinaryCalls/CCLE_MUT_CNA_AMP_DEL_binary_Revealer.txt', delimiter = '\t')
        ccle_mutation_df = ccle_mutation_df.drop(columns=['Description'])
        # print(ccle_mutation_df)
        # COMBINE TWO LISTS [TT_OESOPHAGUS, TT_THYROID]
        tt_oeso = list(ccle_mutation_df['TT_OESOPHAGUS'])
        tt_thy = list(ccle_mutation_df['TT_THYROID'])
        tt_list = []
        for (oeso, thy) in zip(tt_oeso, tt_thy):
            if oeso == 1 or thy == 1: tt_list.append(1)
            else: tt_list.append(0)
        ccle_mutation_df = ccle_mutation_df.drop(columns = ['TT_OESOPHAGUS', 'TT_THYROID'])
        ccle_mutation_df.insert(1, 'TT', tt_list)
        print(ccle_mutation_df)
        # NAME LIST OF DIFFERENT TYPES
        all_list = []
        mut_list = []
        del_list = []
        amp_list = []
        # INDEX LIST OF DIFFERENT TYPES
        mut_idx_list = []
        del_idx_list = []
        amp_idx_list = []
        for row in ccle_mutation_df.itertuples():
            gene_info = row[1].split('_')
            gene = gene_info[0]
            if '.' in gene: 
                gene = gene.split('.')[0]
            if 'MUT' in gene_info[-1]: 
                mut_list.append(gene)
            if 'DEL' in gene_info[-1]:
                del_list.append(gene)
                del_idx_list.append(row[0])
            if 'AMP' in gene_info[-1]:
                amp_list.append(gene)
                amp_idx_list.append(row[0])
            if gene not in all_list:
                all_list.append(gene)
        # # # REMOVE SUFFIX [del]
        ccle_mutation_del_df = ccle_mutation_df.loc[del_idx_list].sort_values(by = ['Name']).reset_index(drop = True)
        # GENE LIST
        ccle_mutation_del_gene_list = list(ccle_mutation_del_df['Name'])
        ccle_mutation_del_gene_maplist = [gene.split('_')[0].upper() for gene in ccle_mutation_del_gene_list]
        ccle_mutation_del_gene_dict = dict(zip(ccle_mutation_del_gene_list, ccle_mutation_del_gene_maplist))
        ccle_mutation_del_df = ccle_mutation_del_df.replace({'Name': ccle_mutation_del_gene_dict})
        # CELL LINE LIST
        ccle_mutation_del_cell_line = list(ccle_mutation_del_df.columns)[1:]
        ccle_mutation_del_cell_maplist = [cell_line.split('_')[0] for cell_line in ccle_mutation_del_cell_line]
        ccle_mutation_del_cell_dict = dict(zip(ccle_mutation_del_cell_line, ccle_mutation_del_cell_maplist))
        ccle_mutation_del_df = ccle_mutation_del_df.rename(columns = ccle_mutation_del_cell_dict) 
        ccle_mutation_del_df.to_csv('../' + dataset + '/init_data/ccle_mutation_del.csv', index = False, header = True)
        print(ccle_mutation_del_df)
        # # # REMOVE SUFFIX [amp]
        # GENE LIST
        ccle_mutation_amp_df = ccle_mutation_df.loc[amp_idx_list].sort_values(by = ['Name']).reset_index(drop = True)
        ccle_mutation_amp_gene_list = list(ccle_mutation_amp_df['Name'])
        ccle_mutation_amp_gene_maplist = [gene.split('_')[0].upper() for gene in ccle_mutation_amp_gene_list]
        ccle_mutation_amp_gene_dict = dict(zip(ccle_mutation_amp_gene_list, ccle_mutation_amp_gene_maplist))
        ccle_mutation_amp_df = ccle_mutation_amp_df.replace({'Name': ccle_mutation_amp_gene_dict})
        # CELL LINE LIST
        ccle_mutation_amp_cell_line = list(ccle_mutation_amp_df.columns)[1:]
        ccle_mutation_amp_cell_maplist = [cell_line.split('_')[0] for cell_line in ccle_mutation_amp_cell_line]
        ccle_mutation_amp_cell_dict = dict(zip(ccle_mutation_amp_cell_line, ccle_mutation_amp_cell_maplist))
        ccle_mutation_amp_df = ccle_mutation_amp_df.rename(columns = ccle_mutation_amp_cell_dict) 
        ccle_mutation_amp_df.to_csv('../' + dataset + '/init_data/ccle_mutation_amp.csv', index = False, header = True)
        print(ccle_mutation_amp_df)

    def kegg(self, dataset):
        kegg_pathway_df = pd.read_csv('../' + dataset + '/raw_data/KEGG/full_kegg_pathway_list.csv')
        kegg_pathway_df = kegg_pathway_df[['source', 'target', 'pathway_name']]
        kegg_df = kegg_pathway_df[kegg_pathway_df['pathway_name'].str.contains('signaling pathway|signaling pathways', case=False)]
        print(kegg_df['pathway_name'].value_counts())
        # import pdb; pdb.set_trace()
        kegg_df = kegg_df.rename(columns={'source': 'src', 'target': 'dest'})
        src_list = list(kegg_df['src'])
        dest_list = list(kegg_df['dest'])
        path_list = list(kegg_df['pathway_name'])
        # ADJUST ALL GENES TO UPPERCASE
        up_src_list = []
        for src in src_list:
            up_src = src.upper()
            up_src_list.append(up_src)
        up_dest_list = []
        for dest in dest_list:
            up_dest = dest.upper()
            up_dest_list.append(up_dest)
        up_kegg_conn_dict = {'src': up_src_list, 'dest': up_dest_list}
        up_kegg_df = pd.DataFrame(up_kegg_conn_dict)
        up_kegg_df = up_kegg_df.drop_duplicates()
        up_kegg_df.to_csv('../' + dataset + '/init_data/up_kegg.csv', index=False, header=True)
        kegg_gene_list = list(set(list(up_kegg_df['src']) + list(up_kegg_df['dest'])))
        print('----- NUMBER OF GENES IN KEGG: ' + str(len(kegg_gene_list)) + ' -----')
        print(up_kegg_df.shape)

        up_kegg_path_conn_dict = {'src': up_src_list, 'dest': up_dest_list, 'path': path_list}
        up_kegg_path_df = pd.DataFrame(up_kegg_path_conn_dict)
        up_kegg_path_df = up_kegg_path_df.drop_duplicates()
        up_kegg_path_df.to_csv('../' + dataset + '/init_data/up_kegg_path.csv', index=False, header=True)
        kegg_gene_list = list(set(list(up_kegg_path_df['src']) + list(up_kegg_path_df['dest'])))
        print('----- NUMBER OF GENES IN KEGG: ' + str(len(kegg_gene_list)) + ' -----')
        print(up_kegg_path_df.shape)

    def biogrid(self, dataset):
        biogrid_df = pd.read_table('../' + dataset + '/raw_data/BioGrid/BIOGRID-ALL-3.5.174.mitab.Symbol.txt', delimiter = '\t')
        eh_list = list(biogrid_df['e_h'])
        et_list = list(biogrid_df['e_t'])
        # ADJUST ALL GENES TO UPPERCASE
        up_eh_list = []
        for eh in eh_list:
            up_eh = eh.upper()
            up_eh_list.append(up_eh)
        up_et_list = []
        for et in et_list:
            up_et = et.upper()
            up_et_list.append(up_et)
        up_biogrid_conn_dict = {'e_h': up_eh_list, 'e_t': up_et_list}
        up_biogrid_df = pd.DataFrame(up_biogrid_conn_dict)
        print(up_biogrid_df)
        up_biogrid_df.to_csv('../' + dataset + '/init_data/up_biogrid.csv', index = False, header = True)

    def string(self, dataset):
        string_df = pd.read_csv('../' + dataset + '/raw_data/STRING/9606.protein.links.detailed.v11.0_sym.csv', low_memory=False)
        src_list = list(string_df['Source'])
        tar_list = list(string_df['Target'])
        # ADJUST ALL GENES TO UPPERCASE
        up_src_list = []
        for src in src_list:
            up_src = src.upper()
            up_src_list.append(up_src)
        up_tar_list = []
        for tar in tar_list:
            up_tar = tar.upper()
            up_tar_list.append(up_tar)
        up_string_conn_dict = {'Source': up_src_list, 'Target': up_tar_list}
        up_string_df = pd.DataFrame(up_string_conn_dict)
        print(up_string_df)
        up_string_df.to_csv('../' + dataset + '/init_data/up_string.csv', index = False, header = True)

    def drugbank(self, dataset):
        # INITIALIZE THE DRUG BANK INTO [.csv] FILE
        drugbank_df = pd.read_table('../' + dataset + '/raw_data/DrugBank/drug_tar_drugBank_all.txt', delimiter='\t')
        drug_list = list(set(list(drugbank_df['Drug'])))
        target_gene_list = list(set(list(drugbank_df['Target'])))
        print('----- NUMBER OF DRUGS IN DrugBank: ' + str(len(drug_list)) + ' -----')
        print('----- NUMBER OF GENES IN DrugBank: ' + str(len(target_gene_list)) + ' -----')
        drugbank_df.to_csv('../' + dataset + '/init_data/drugbank.csv', index=False, header=True)



def init_parse(dataset):
    if dataset == 'datainfo-nci':
        ReadFile().combo_input(dataset)
    elif dataset == 'datainfo-oneil':
        ReadFile().parse_drugcomb_fi(dataset)
    # ReadFile().gdsc_rnaseq(dataset)
    # ReadFile().gdsc_cnv(dataset)
    # ReadFile().ccle_meth(dataset)
    # ReadFile().ccle_mutation(dataset)
    # ReadFile().kegg(dataset)
    # ReadFile().biogrid(dataset)
    # ReadFile().string(dataset)
    # ReadFile().drugbank(dataset)

if __name__ == "__main__":
    ### MODEL SELECTION
    # dataset = 'datainfo-nci'
    dataset = 'datainfo-oneil'
    init_parse(dataset)