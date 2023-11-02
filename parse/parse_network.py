import os
import re
import numpy as np
import pandas as pd

'''
Construct network using number (index) with:
(1) KEGG gene-gene interaction
(2) DrugBank drug-gene interaction
'''
class Network():
    def __init__(self):
        pass
    
    def num_kegg_gene_interaction(self, dataset):
        # FORM [gene_num_dict] FOR [KEGG] AND REPLACE
        kegg_gene_annotation_df = pd.read_csv('../' +dataset + '/filtered_data/kegg_gene_annotation.csv')
        kegg_gene_annotation_df['gene_num'] = list(range(1, kegg_gene_annotation_df.shape[0]+1))
        kegg_gene_annotation_df.to_csv('../' +dataset + '/filtered_data/kegg_gene_num_dict.csv', index=False, header=True)
        kegg_gene_num_dict = dict(zip(kegg_gene_annotation_df.kegg_gene, kegg_gene_annotation_df.gene_num))
        kegg_gene_interaction_df = pd.read_csv('../' +dataset + '/filtered_data/kegg_gene_interaction.csv')
        kegg_gene_interaction_df = kegg_gene_interaction_df.replace({'src': kegg_gene_num_dict, 'dest': kegg_gene_num_dict})
        kegg_gene_interaction_df.to_csv('../' +dataset + '/filtered_data/kegg_gene_num_interaction.csv', index=False, header=True)
        kegg_gene_interaction_inv_df = kegg_gene_interaction_df[['dest', 'src']]
        kegg_gene_interaction_inv_df = kegg_gene_interaction_inv_df.rename(columns={'dest': 'src', 'src': 'dest'})
        kegg_gene_interaction_sym_df = pd.concat([kegg_gene_interaction_df, kegg_gene_interaction_inv_df], ignore_index=True)
        kegg_gene_interaction_sym_df.to_csv('../' +dataset + '/filtered_data/kegg_gene_num_interaction_sym.csv', index=False, header=True)

    def num_drugbank_interaction(self, dataset):
        # FORM [drug_num_dict] FOR [DRUG BANK] AND REPLACE
        kegg_gene_num_dict_df =pd.read_csv('../' +dataset + '/filtered_data/kegg_gene_num_dict.csv')
        kegg_gene_num_dict = dict(zip(kegg_gene_num_dict_df.kegg_gene, kegg_gene_num_dict_df.gene_num))
        final_drugbank_df = pd.read_csv('../' +dataset + '/filtered_data/final_drugbank.csv')
        drugbank_drug_annotation_list = sorted(list(set(list(final_drugbank_df['Drug']))))
        drugbank_drug_annotation_df = pd.DataFrame(data=drugbank_drug_annotation_list, columns=['Drug'])
        drugbank_drug_annotation_df['drug_num'] = list(range(kegg_gene_num_dict_df.shape[0]+1, \
                                                kegg_gene_num_dict_df.shape[0]+1+drugbank_drug_annotation_df.shape[0]))
        # import pdb; pdb.set_trace()
        drugbank_drug_annotation_df.to_csv('../' +dataset + '/filtered_data/drug_num_dict.csv', index=False, header=True)
        drugbank_drug_num_dict = dict(zip(drugbank_drug_annotation_df.Drug, drugbank_drug_annotation_df.drug_num))
        final_drugbank_num_df = final_drugbank_df.replace({'Drug': drugbank_drug_num_dict, 'Target': kegg_gene_num_dict})
        final_drugbank_num_df.to_csv('../' +dataset + '/filtered_data/final_drugbank_num.csv', index=False, header=True)
        final_drugbank_num_inv_df = final_drugbank_num_df[['Target', 'Drug']]
        final_drugbank_num_inv_df = final_drugbank_num_inv_df.rename(columns={'Target': 'Drug', 'Drug': 'Target'})
        final_drugbank_num_sym_df = pd.concat([final_drugbank_num_df, final_drugbank_num_inv_df], ignore_index=True)
        final_drugbank_num_sym_df.to_csv('../' +dataset + '/filtered_data/final_drugbank_num_sym.csv', index=False, header=True)
    
    def combine_network(self, dataset):
        kegg_gene_num_interaction_df = pd.read_csv('../' +dataset + '/filtered_data/kegg_gene_num_interaction.csv')
        kegg_gene_num_interaction_df = kegg_gene_num_interaction_df.rename({'src': 'from', 'dest': 'to'}, axis='columns')
        final_drugbank_num_df = pd.read_csv('../' +dataset + '/filtered_data/final_drugbank_num.csv')
        final_drugbank_num_df = final_drugbank_num_df.rename({'Drug': 'from', 'Target': 'to'}, axis='columns')
        all_edge_num_df = pd.concat([kegg_gene_num_interaction_df, final_drugbank_num_df])
        all_edge_num_df.to_csv('../' +dataset + '/filtered_data/all_edge_num.csv', index=False, header=True)


### DATASET SELECTION
# dataset = 'datainfo-nci'
dataset = 'datainfo-oneil'
Network().num_kegg_gene_interaction(dataset)
Network().num_drugbank_interaction(dataset)
Network().combine_network(dataset)