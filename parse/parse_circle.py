import os
import re
import numpy as np
import pandas as pd

from lxml import etree
from pubchempy import *


'''
Pin the number of cell lines by following tables:
(1) DeepLearningInput.csv
(2) GDSC RNA-Seq
(3) GDSC CNV
(4) CCLE Meth
'''
class CellLineAnnotation():
    def __init__(self):
        pass

    def parse_cell_xml(self, dataset):
        # # # INTIALIZE THE CELL LINE ANNOTATION
        cellosaurus_doc = etree.parse('../' + dataset +'/raw_data/Annotation/cellosaurus.xml')
        count = 0
        accession_list = []
        identifier_list = []
        synonym_list = []
        species_list = []
        # KEEP ONLY ['Homo Sapiens'] CELL LINE
        for att in cellosaurus_doc.xpath('//cell-line'):
            count += 1
            print('-------------', count)
            accession = att.xpath('.//accession[@type="primary"]/text()')[0]
            identifier = att.xpath('.//name[@type="identifier"]/text()')[0]
            synonym = att.xpath('.//name[@type="synonym"]/text()')
            species = att.xpath('.//cv-term[@terminology="NCBI-Taxonomy"]/text()')
            if 'Homo sapiens' not in species:
                print(species)
                continue
            accession_list.append(accession)
            identifier_list.append(identifier)
            synonym_list.append(synonym)
            species_list.append(species)
        cell_annotation_df = pd.DataFrame({'Accession': accession_list,
                                           'Identifier': identifier_list,
                                           'Synonym': synonym_list,
                                           'Species': species_list})
        cell_annotation_df.to_csv('../' + dataset +'/init_data/cell_annotation.csv', index=False, header=True)

    def dl_cell_annotation(self, dataset):
        # # # REPLACE ALL CELL LINE NAME WITH [Accession]
        cell_annotation_df = pd.read_csv('../' + dataset +'/init_data/cell_annotation_manual.csv', keep_default_na=False)
        ### [dl_input.csv]
        if dataset == 'datainfo-nci':
            dl_input_df = pd.read_csv('../' + dataset +'/init_data/almanac_dl_input.csv')
        elif dataset == 'datainfo-oneil':
            dl_input_df = pd.read_csv('../' + dataset +'/init_data/oneil_dl_input.csv')
        dl_input_cell_line_list = sorted(list(set(list(dl_input_df['Cell Line Name']))))
        dl_cell_accession_list = []
        dl_cell_identifier_list = []
        dl_cell_synonym_list = []
        for dl_cell in dl_input_cell_line_list:
            for row in cell_annotation_df.itertuples():
                if (dl_cell == row.Identifier) or (dl_cell in eval(row.Synonym)):
                    dl_cell_accession_list.append(row.Accession)
                    dl_cell_identifier_list.append(row.Identifier)
                    dl_cell_synonym_list.append(eval(row.Synonym))
        dl_cell_annotation_df = pd.DataFrame({'dl_cell': dl_input_cell_line_list,
                                              'Accession': dl_cell_accession_list,
                                              'Identifier': dl_cell_identifier_list,
                                              'Synonym': dl_cell_synonym_list})
        dl_cell_annotation_df.to_csv('../' + dataset +'/init_data/dl_cell_annotation.csv', index=False, header=True)
        
    def omics_cell(self, dataset):
        # # # [dl_cell_annotation]
        dl_cell_annotation_df = pd.read_csv('../' + dataset +'/init_data/dl_cell_annotation.csv', keep_default_na=False)
        ### [CCLE Methylation]
        max_cmeth_df = pd.read_csv('../' + dataset +'/init_data/ccle_methylation_max.csv', low_memory=False)
        cmeth_cell_line_list = sorted(list(max_cmeth_df.columns)[1:])
        cmeth_selected_cell_list = []
        cmeth_cell_accession_list = []
        for cmeth_cell in cmeth_cell_line_list:
            for row in dl_cell_annotation_df.itertuples():
                if (cmeth_cell == row.Identifier) or (cmeth_cell in eval(row.Synonym)):
                    cmeth_selected_cell_list.append(cmeth_cell)
                    cmeth_cell_accession_list.append(row.Accession)
        cmeth_cell_accession_df = pd.DataFrame({'cmeth_cell': cmeth_selected_cell_list,
                                                'Accession': cmeth_cell_accession_list})  
        dl_cmeth_cell_annotation_df = pd.merge(dl_cell_annotation_df, cmeth_cell_accession_df, \
                            how='left', left_on='Accession', right_on='Accession')
        dl_cmeth_cell_annotation_df = dl_cmeth_cell_annotation_df.dropna().reset_index(drop=True)
        # # # [GDSC RNASeq]
        rna_df = pd.read_csv('../' + dataset +'/init_data/gdsc_rnaseq_manual.csv', low_memory=False)
        rna_cell_line_list = sorted(list(rna_df.columns)[1:])
        rna_selected_cell_list = []
        rna_cell_accession_list = []
        for rna_cell in rna_cell_line_list:
            for row in dl_cmeth_cell_annotation_df.itertuples():
                if (rna_cell == row.Identifier) or (rna_cell in eval(row.Synonym)):
                    rna_selected_cell_list.append(rna_cell)
                    rna_cell_accession_list.append(row.Accession)
        rna_cell_accession_df = pd.DataFrame({'rna_cell': rna_selected_cell_list,
                                              'Accession': rna_cell_accession_list})  
        dl_cmeth_rna_cell_annotation_df = pd.merge(dl_cmeth_cell_annotation_df, rna_cell_accession_df, \
                            how='left', left_on='Accession', right_on='Accession')
        dl_cmeth_rna_cell_annotation_df = dl_cmeth_rna_cell_annotation_df.dropna().reset_index(drop=True)
        # # # [GDSC CNV]
        cnv_df = pd.read_csv('../' + dataset +'/init_data/gdsc_cnv_manual.csv', low_memory=False)
        cnv_cell_line_list = sorted(list(cnv_df.columns)[1:])
        cnv_selected_cell_list = []
        cnv_cell_accession_list = []
        for cnv_cell in cnv_cell_line_list:
            for row in dl_cmeth_cell_annotation_df.itertuples():
                if (cnv_cell == row.Identifier) or (cnv_cell in eval(row.Synonym)):
                    cnv_selected_cell_list.append(cnv_cell)
                    cnv_cell_accession_list.append(row.Accession)
        cnv_cell_accession_df = pd.DataFrame({'cnv_cell': cnv_selected_cell_list,
                                              'Accession': cnv_cell_accession_list})
        dl_cmeth_rna_cnv_cell_annotation_df = pd.merge(dl_cmeth_rna_cell_annotation_df, cnv_cell_accession_df, \
                            how='left', left_on='Accession', right_on='Accession')
        dl_cmeth_rna_cnv_cell_annotation_df = dl_cmeth_rna_cnv_cell_annotation_df.dropna().reset_index(drop=True)
        # # # [CCLE Mutation]
        cmut_amp_df = pd.read_csv('../' + dataset +'/init_data/ccle_mutation_amp.csv')
        cmut_amp_cell_line_list = sorted(list(cmut_amp_df.columns)[1:])
        cmut_amp_selected_cell_list = []
        cmut_amp_cell_accession_list = []
        for cmut_amp_cell in cmut_amp_cell_line_list:
            for row in dl_cmeth_cell_annotation_df.itertuples():
                if (cmut_amp_cell == row.Identifier) or (cmut_amp_cell in eval(row.Synonym)):
                    cmut_amp_selected_cell_list.append(cmut_amp_cell)
                    cmut_amp_cell_accession_list.append(row.Accession)
        cmut_amp_cell_accession_df = pd.DataFrame({'cmut_amp_cell': cmut_amp_selected_cell_list,
                                              'Accession': cmut_amp_cell_accession_list})
        dl_cmeth_rna_cnv_cmut_amp_cell_annotation_df = pd.merge(dl_cmeth_rna_cnv_cell_annotation_df, cmut_amp_cell_accession_df, \
                            how='left', left_on='Accession', right_on='Accession')
        dl_cmeth_rna_cnv_cmut_amp_cell_annotation_df = dl_cmeth_rna_cnv_cmut_amp_cell_annotation_df.dropna().reset_index(drop=True)
        dl_cmeth_rna_cnv_cmut_amp_cell_annotation_df.to_csv('../' + dataset +'/init_data/omics_cell_annotation.csv', index=False, header=True)  
        print(dl_cmeth_rna_cnv_cmut_amp_cell_annotation_df)

    def tail_cell(self, dataset):
        # READ [omics_cell_annotation]
        omics_cell_df = pd.read_csv('../' + dataset +'/init_data/omics_cell_annotation.csv')
        # TAIL [NCI ALMANAC] CELL LINEs
        new_dl_input_cell_line_list = list(omics_cell_df['dl_cell'])
        ### [dl_input.csv]
        if dataset == 'datainfo-nci':
            dl_input_df = pd.read_csv('../' + dataset +'/init_data/almanac_dl_input.csv')
        elif dataset == 'datainfo-oneil':
            dl_input_df = pd.read_csv('../' + dataset +'/init_data/oneil_dl_input.csv')
        tail_cell_dl_input_df = dl_input_df[dl_input_df['Cell Line Name'].isin(new_dl_input_cell_line_list)].reset_index(drop=True)
        tail_cell_dl_input_df.to_csv('../' + dataset +'/mid_cell_line/tail_cell_dl_input.csv', index=False, header=True)
        print(tail_cell_dl_input_df)
        # TAIL [GDSC RNA-Seq] CELL LINEs // KEEP [RNA-Seq] CELL LINE NAMES CONSISTENT WITH [NCI ALMANAC]
        omics_rna_cell_line_list = list(omics_cell_df['rna_cell'])
        rna_df = pd.read_csv('../' + dataset +'/init_data/gdsc_rnaseq_manual.csv', low_memory=False)
        tail_cell_rna_df = rna_df[omics_rna_cell_line_list]
        nci_rna_cell_line_dict = dict(zip(omics_cell_df.rna_cell, omics_cell_df.dl_cell))
        tail_cell_rna_df = tail_cell_rna_df.rename(columns=nci_rna_cell_line_dict)
        tail_cell_rna_df.insert(0, 'symbol', list(rna_df['symbol']))
        tail_cell_rna_df.to_csv('../' + dataset +'/mid_cell_line/tail_cell_rna.csv', index=False, header=True)
        print(tail_cell_rna_df)
        # TAIL [GDSC CNV] CELL LINEs // KEEP [CNV] CELL LINE NAMES CONSISTENT WITH [NCI ALMANAC]
        omics_cnv_cell_line_list = list(omics_cell_df['cnv_cell'])
        cnv_df = pd.read_csv('../' + dataset +'/init_data/gdsc_cnv_manual.csv', low_memory=False)
        tail_cell_cnv_df = cnv_df[omics_cnv_cell_line_list]
        nci_cnv_cell_line_dict = dict(zip(omics_cell_df.cnv_cell, omics_cell_df.dl_cell))
        tail_cell_cnv_df = tail_cell_cnv_df.rename(columns=nci_cnv_cell_line_dict)
        tail_cell_cnv_df.insert(0, 'symbol', list(cnv_df['symbol']))
        tail_cell_cnv_df = tail_cell_cnv_df.reset_index(drop=True)
        tail_cell_cnv_df.to_csv('../' + dataset +'/mid_cell_line/tail_cell_cnv.csv', index=False, header=True)
        print(tail_cell_cnv_df)
        # TAIL [CCLE Methylation] // KEEP [CMeth] CELL LINE NAMES CONSISTENT WITH [NCI ALMANAC]
        omics_cmeth_cell_line_list = list(omics_cell_df['cmeth_cell'])
        # MAX [CCLE Methylation]
        max_cmeth_df = pd.read_csv('../' + dataset +'/init_data/ccle_methylation_max.csv', low_memory=False)
        tail_cell_max_cmeth_df = max_cmeth_df[omics_cmeth_cell_line_list]
        nci_cmeth_cell_line_dict = dict(zip(omics_cell_df.cmeth_cell, omics_cell_df.dl_cell))
        tail_cell_max_cmeth_df = tail_cell_max_cmeth_df.rename(columns=nci_cmeth_cell_line_dict)
        tail_cell_max_cmeth_df.insert(0, 'locus_id', list(max_cmeth_df['locus_id']))
        tail_cell_max_cmeth_df.to_csv('../' + dataset +'/mid_cell_line/tail_cell_cmeth_max.csv', index=False, header=True)
        print(tail_cell_max_cmeth_df)
        # MIN [CCLE Methylation]
        min_cmeth_df = pd.read_csv('../' + dataset +'/init_data/ccle_methylation_min.csv', low_memory=False)
        tail_cell_min_cmeth_df = min_cmeth_df[omics_cmeth_cell_line_list]
        nci_cmeth_cell_line_dict = dict(zip(omics_cell_df.cmeth_cell, omics_cell_df.dl_cell))
        tail_cell_min_cmeth_df = tail_cell_min_cmeth_df.rename(columns=nci_cmeth_cell_line_dict)
        tail_cell_min_cmeth_df.insert(0, 'locus_id', list(min_cmeth_df['locus_id']))
        tail_cell_min_cmeth_df.to_csv('../' + dataset +'/mid_cell_line/tail_cell_cmeth_min.csv', index=False, header=True)
        print(tail_cell_min_cmeth_df)
        # TAIL [CCLE AMP/DEL]
        # [amp]
        omics_mut_cell_line_list = list(omics_cell_df['cmut_amp_cell'])
        cmut_amp_df = pd.read_csv('../' + dataset +'/init_data/ccle_mutation_amp.csv')
        tail_cell_cmut_amp_df = cmut_amp_df[omics_mut_cell_line_list]
        nci_cmut_amp_cell_line_dict = dict(zip(omics_cell_df.cmut_amp_cell, omics_cell_df.dl_cell))
        tail_cell_cmut_amp_df = tail_cell_cmut_amp_df.rename(columns=nci_cmut_amp_cell_line_dict)
        tail_cell_cmut_amp_df.insert(0, 'Name', list(cmut_amp_df['Name']))
        tail_cell_cmut_amp_df.to_csv('../' + dataset +'/mid_cell_line/tail_cell_cmut_amp.csv', index=False, header=True)
        print(tail_cell_cmut_amp_df)
        # [del]
        cmut_del_df = pd.read_csv('../' + dataset +'/init_data/ccle_mutation_del.csv')
        tail_cell_cmut_del_df = cmut_del_df[omics_mut_cell_line_list]
        tail_cell_cmut_del_df = tail_cell_cmut_del_df.rename(columns=nci_cmut_amp_cell_line_dict)
        tail_cell_cmut_del_df.insert(0, 'Name', list(cmut_del_df['Name']))
        tail_cell_cmut_del_df.to_csv('../' + dataset +'/mid_cell_line/tail_cell_cmut_del.csv', index=False, header=True)
        print(tail_cell_cmut_del_df)

'''
Pin the number of genes from following intersection of tables:
(1) KEGG ['src', 'dest']
(2) GDSC RNASeq ['symbol']
(3) DrugBank ['Target']
'''
class GeneAnnotation():
    def __init__(self):
        pass

    def check_gdsc_cnv_tail_overzero(self, dataset):
        tail_cell_cnv_df = pd.read_csv('../' + dataset +'/mid_cell_line/tail_cell_cnv.csv')
        tail_cell_cnv_df = tail_cell_cnv_df.replace(['missing'], 0.0)
        threshold = len(tail_cell_cnv_df.columns[1:]) * 0.9
        tail_cell_cnv_gene_deletion_list = [row[0] for row in tail_cell_cnv_df.itertuples() if list(row[2:]).count(0)>threshold]
        print(len(tail_cell_cnv_gene_deletion_list))

    def check_gdsc_rna_tail_overzero(self, dataset):
        tail_cell_rna_df = pd.read_csv('../' + dataset +'/mid_cell_line/tail_cell_rna.csv')
        tail_cell_rna_df = tail_cell_rna_df.replace(['missing'], 0.0)
        threshold = len(tail_cell_rna_df.columns[1:]) * 0.9
        tail_cell_rna_gene_deletion_list = [row[0] for row in tail_cell_rna_df.itertuples() if list(row[2:]).count(0)>threshold]
        print(len(tail_cell_rna_gene_deletion_list))

    def kegg_omics_intersect(self, dataset):
        # # GET [2241] KEGG PATHWAY GENES
        kegg_df = pd.read_csv('../' + dataset +'/init_data/up_kegg.csv')
        kegg_gene_list = sorted(list(set(list(kegg_df['src']) + list(kegg_df['dest']))))
        print(len(kegg_gene_list))
        # GET [OMICS] INTERSETED GENES
        tail_cell_rna_df = pd.read_csv('../' + dataset +'/mid_cell_line/tail_cell_rna.csv', low_memory=False)
        tail_cell_rna_df = tail_cell_rna_df.sort_values(by=['symbol'])
        rna_gene_list = list(tail_cell_rna_df['symbol'])
        tail_cell_cnv_df = pd.read_csv('../' + dataset +'/mid_cell_line/tail_cell_cnv.csv', low_memory=False)
        tail_cell_cnv_df = tail_cell_cnv_df.sort_values(by=['symbol'])
        cnv_gene_list = list(tail_cell_cnv_df['symbol'])
        tail_cell_max_cmeth_df = pd.read_csv('../' + dataset +'/mid_cell_line/tail_cell_cmeth_max.csv', low_memory=False)
        tail_cell_max_cmeth_df = tail_cell_max_cmeth_df.sort_values(by=['locus_id'])
        tail_cell_min_cmeth_df = pd.read_csv('../' + dataset +'/mid_cell_line/tail_cell_cmeth_min.csv', low_memory=False)
        tail_cell_min_cmeth_df = tail_cell_min_cmeth_df.sort_values(by=['locus_id'])
        cmeth_gene_list = list(tail_cell_max_cmeth_df['locus_id'])
        tail_cell_cmut_amp_df = pd.read_csv('../' + dataset +'/mid_cell_line/tail_cell_cmut_amp.csv')
        tail_cell_cmut_amp_df = tail_cell_cmut_amp_df.sort_values(by=['Name'])
        cmut_amp_gene_list = list(tail_cell_cmut_amp_df['Name'])
        tail_cell_cmut_del_df = pd.read_csv('../' + dataset +'/mid_cell_line/tail_cell_cmut_del.csv')
        tail_cell_cmut_del_df = tail_cell_cmut_del_df.sort_values(by=['Name'])
        cmut_del_gene_list = list(tail_cell_cmut_del_df['Name'])
        omics_gene_set = set(rna_gene_list).intersection(set(cnv_gene_list)).intersection(set(cmeth_gene_list))\
                            .intersection(set(cmut_amp_gene_list)).intersection(set(cmut_del_gene_list))
        omics_gene_list = sorted(list(omics_gene_set))
        # # # LEFT JOIN TO AUTO MAP
        kegg_gene_df = pd.DataFrame(data=kegg_gene_list, columns=['kegg_gene'])
        omics_gene_df = pd.DataFrame(data=omics_gene_list, columns=['omics_gene'])
        kegg_omics_gene_df = pd.merge(kegg_gene_df, omics_gene_df, how='left', left_on='kegg_gene', right_on='omics_gene')
        kegg_omics_gene_df = kegg_omics_gene_df.dropna().reset_index(drop=True) # [2241] => [] GENEs
        # TAIL [KEGG] GENE => [1489] GENES
        kegg_gene_deletion_list = [gene for gene in kegg_gene_list if gene not in list(kegg_omics_gene_df['kegg_gene'])]
        kegg_gene_deletion_index = [row[0] for row in kegg_df.itertuples() \
                                    if row[1] in kegg_gene_deletion_list or row[2] in kegg_gene_deletion_list]
        new_kegg_df = kegg_df.drop(kegg_gene_deletion_index).reset_index(drop=True)
        new_kegg_df = new_kegg_df.sort_values(by=['src', 'dest'])
        new_kegg_df.to_csv('../' + dataset +'/filtered_data/kegg_gene_interaction.csv', index=False, header=True)
        new_kegg_gene_list = sorted(list(set(list(new_kegg_df['src']) + list(new_kegg_df['dest']))))
        new_kegg_gene_df = pd.DataFrame(data=new_kegg_gene_list, columns=['kegg_gene'])
        new_kegg_gene_df.to_csv('../' + dataset +'/filtered_data/kegg_gene_annotation.csv', index=False, header=True)
        # TAIL [GDSC RNASeq] => [1489] GENES
        tail_cell_gene_rna_df = tail_cell_rna_df[tail_cell_rna_df['symbol'].isin(new_kegg_gene_list)].reset_index(drop=True)
        tail_cell_gene_rna_df.to_csv('../' + dataset +'/mid_gene/tail_cell_gene_rna.csv', index=False, header=True)
        print(tail_cell_gene_rna_df)
        # TAIL [GDSC CNV] => [1489] GENES
        tail_cell_gene_cnv_df = tail_cell_cnv_df[tail_cell_cnv_df['symbol'].isin(new_kegg_gene_list)].reset_index(drop=True)
        tail_cell_gene_cnv_df.to_csv('../' + dataset +'/mid_gene/tail_cell_gene_cnv.csv', index=False, header=True)
        print(tail_cell_gene_cnv_df)
        # TAIL [CCLE Methylation] => [1489] GENES
        tail_cell_gene_max_cmeth_df = tail_cell_max_cmeth_df[tail_cell_max_cmeth_df['locus_id'].isin(new_kegg_gene_list)].reset_index(drop=True)
        tail_cell_gene_max_cmeth_df.to_csv('../' + dataset +'/mid_gene/tail_cell_gene_cmeth_max.csv', index=False, header=True)
        print(tail_cell_gene_max_cmeth_df)
        # TAIL [CCLE Methylation] => [1489] GENES
        tail_cell_gene_min_cmeth_df = tail_cell_min_cmeth_df[tail_cell_min_cmeth_df['locus_id'].isin(new_kegg_gene_list)].reset_index(drop=True)
        tail_cell_gene_min_cmeth_df.to_csv('../' + dataset +'/mid_gene/tail_cell_gene_cmeth_min.csv', index=False, header=True)
        print(tail_cell_gene_min_cmeth_df)
        # TAIL [CCLE MUTATION AMP] => [1489] GENES
        tail_cell_gene_cmut_amp_df = tail_cell_cmut_amp_df[tail_cell_cmut_amp_df['Name'].isin(new_kegg_gene_list)].reset_index(drop=True)
        tail_cell_gene_cmut_amp_df.to_csv('../' + dataset +'/mid_gene/tail_cell_gene_cmut_amp.csv', index=False, header=True)
        print(tail_cell_gene_cmut_amp_df)
        # TAIL [CCLE MUTATION DEL] => [1489] GENES
        tail_cell_gene_cmut_del_df = tail_cell_cmut_del_df[tail_cell_cmut_del_df['Name'].isin(new_kegg_gene_list)].reset_index(drop=True)
        tail_cell_gene_cmut_del_df.to_csv('../' + dataset +'/mid_gene/tail_cell_gene_cmut_del.csv', index=False, header=True)
        print(tail_cell_gene_cmut_del_df)

    def kegg_path_omics_intersect(self, dataset):
        kegg_gene_annotation_df = pd.read_csv('../' + dataset +'/filtered_data/kegg_gene_annotation.csv')
        new_kegg_gene_list = list(kegg_gene_annotation_df['kegg_gene'])
        kegg_path_df = pd.read_csv('../' + dataset +'/init_data/up_kegg_path.csv')
        kegg_path_gene_list = list(set(list(kegg_path_df['src'])+list(kegg_path_df['dest'])))
        # import pdb; pdb.set_trace()
        # TAIL [kegg_path_df] TO NEW GENES [1489] GENES
        kegg_path_gene_deletion_list = [gene for gene in kegg_path_gene_list if gene not in new_kegg_gene_list]
        kegg_path_gene_deletion_index = [row[0] for row in kegg_path_df.itertuples() \
                                    if row[1] in kegg_path_gene_deletion_list or row[2] in kegg_path_gene_deletion_list]
        new_kegg_path_df = kegg_path_df.drop(kegg_path_gene_deletion_index).reset_index(drop=True)
        new_kegg_path_df = new_kegg_path_df.sort_values(by=['path', 'src', 'dest'])
        new_kegg_path_df = new_kegg_path_df.drop_duplicates()
        new_kegg_path_df.to_csv('../' + dataset +'/filtered_data/kegg_path_gene_interaction.csv', index=False, header=True)
        # new_kegg_path_gene_list = list(set(list(new_kegg_path_df['src'])+list(new_kegg_path_df['dest'])))
        # print(len(new_kegg_path_gene_list))


    def kegg_drugbank_gene_intersect(self, dataset):
        # TAIL [DrugBank] GENE
        kegg_gene_annotation_df = pd.read_csv('../' + dataset +'/filtered_data/kegg_gene_annotation.csv')
        kegg_gene_list = list(kegg_gene_annotation_df['kegg_gene'])
        drugbank_df = pd.read_csv('../' + dataset +'/init_data/drugbank.csv')
        new_drugbank_df = drugbank_df[drugbank_df['Target'].isin(kegg_gene_list)].reset_index(drop=True)
        new_drugbank_df.to_csv('../' + dataset +'/mid_gene/tail_gene_drugbank.csv', index=False, header=True)


'''
Pin the number of drug from following tables:
(1) tail_gene_drugbank.csv
(2) DeepLearningInput.csv
'''
class DrugAnnotation():
    def __init__(self):
        pass

    def nci_drugbank_drug_intersect(self, dataset):
        tail_cell_dl_input_df = pd.read_csv('../' + dataset +'/mid_cell_line/tail_cell_dl_input.csv')
        tail_cell_dl_input_drug_list = sorted(list(set(list(tail_cell_dl_input_df['Drug A']) + list(tail_cell_dl_input_df['Drug B']))))
        tail_gene_drugbank_df = pd.read_csv('../' + dataset +'/mid_gene/tail_gene_drugbank.csv')
        tail_gene_drugbank_drug_list = sorted(list(set(list(tail_gene_drugbank_df['Drug']))))
        # # # LEFT JOIN [NCI ALMANAC]
        dl_input_drug_uprmv_list = [drug.replace('-', '').replace('_', '').upper() for drug in tail_cell_dl_input_drug_list]
        dl_input_drug_uprmv_df = pd.DataFrame({'input_drug': tail_cell_dl_input_drug_list, 'input_uprmv': dl_input_drug_uprmv_list})
        drugbank_uprmv_druglist = []
        for drug in tail_gene_drugbank_drug_list:
            tmp_drug = drug.replace('-', '')
            tmp_drug = tmp_drug.replace('(', '').replace(')', '')
            tmp_drug = tmp_drug.replace('[', '').replace(']', '')
            tmp_drug = tmp_drug.replace('{', '').replace('}', '')
            tmp_drug = tmp_drug.replace(',', '').upper()
            drugbank_uprmv_druglist.append(tmp_drug)
        drugbank_uprmv_drug_df = pd.DataFrame({'drugbank_uprmv': drugbank_uprmv_druglist, 'drugbank_drug': tail_gene_drugbank_drug_list})
        nci_drugbank_drug_df = pd.merge(dl_input_drug_uprmv_df, drugbank_uprmv_drug_df, how='left', left_on='input_uprmv', right_on='drugbank_uprmv')
        nci_drugbank_drug_df = nci_drugbank_drug_df.dropna().reset_index(drop=True)
        # TAIL [NCI ALMANAC] DRUGS
        dl_input_drug_deletion_list = list(set([drug for drug in tail_cell_dl_input_drug_list if drug not in list(nci_drugbank_drug_df['input_drug'])]))
        dl_input_drug_deletion_index = [row[0] for row in tail_cell_dl_input_df.itertuples() \
                                    if row[1] in dl_input_drug_deletion_list or row[2] in dl_input_drug_deletion_list]
        tail_cell_drug_dl_input_df = tail_cell_dl_input_df.drop(dl_input_drug_deletion_index).reset_index(drop=True)
        tail_cell_drug_dl_input_df.to_csv('../' + dataset +'/mid_drug/tail_cell_drug_dl_input.csv', index=False, header=True)
        # TAIL [DrugBank] DRUGS // REPLACE [DrugBank] DRUGs CONSISTENT WITH [NCI ALMANAC]
        tail_gene_drug_drugbank_df = tail_gene_drugbank_df[tail_gene_drugbank_df['Drug']\
                                    .isin(list(nci_drugbank_drug_df['drugbank_drug']))].reset_index(drop=True)
        nci_drugbank_drug_dict = dict(zip(nci_drugbank_drug_df.drugbank_drug, nci_drugbank_drug_df.input_drug))
        tail_gene_drug_drugbank_df = tail_gene_drug_drugbank_df.replace({'Drug': nci_drugbank_drug_dict})
        tail_gene_drug_drugbank_df.to_csv('../' + dataset +'/mid_drug/tail_gene_drug_drugbank.csv', index=False, header=True)
        # print(nci_drugbank_drug_df)

'''
Recheck the number of cell lines after tailing drugs in NCI ALMANAC:
(1) tail_cell_drug_dl_input.csv
(2) tail_cell_gene_rna.csv
'''
class RecheckFinal():
    def __init__(self):
        pass
    
    def recheck_cell_line(self, dataset):
        # TAILED [DeepLearning]
        tail_cell_drug_dl_input_df = pd.read_csv('../' + dataset +'/mid_drug/tail_cell_drug_dl_input.csv')
        tail_cell_drug_dl_input_cell_list = list(set(list(tail_cell_drug_dl_input_df['Cell Line Name'])))
        # TAILED [GDSC RNA_seq]
        tail_cell_gene_rna_df = pd.read_csv('../' + dataset +'/mid_gene/tail_cell_gene_rna.csv')
        tail_cell_gene_rna_cell_list = tail_cell_gene_rna_df.columns[1:]
        recheck_cell_line = [cell_line for cell_line in tail_cell_gene_rna_cell_list if cell_line not in tail_cell_drug_dl_input_cell_list]
        if len(recheck_cell_line)==0 : print('NO MORE CHECK NEEDED')

    def final(self, dataset):
        # [DeepLearningInput]
        tail_cell_drug_dl_input_df = pd.read_csv('../' + dataset +'/mid_drug/tail_cell_drug_dl_input.csv')
        # tail_cell_drug_dl_input_df['Score'] = (tail_cell_drug_dl_input_df['Score'] - tail_cell_drug_dl_input_df['Score'].mean()) / tail_cell_drug_dl_input_df['Score'].std()    
        tail_cell_drug_dl_input_df.to_csv('../' + dataset +'/filtered_data/final_dl_input.csv', index=False, header=True)
        cell_line_list = sorted(list(set(tail_cell_drug_dl_input_df['Cell Line Name'])))
        print('----- NUMBER OF CELL LINES IN DEEP LEARNING INPUT: ' + str(len(cell_line_list)) + ' ------')
        print(cell_line_list)
        # [RNA-Seq]
        tail_cell_gene_rna_df = pd.read_csv('../' + dataset +'/mid_gene/tail_cell_gene_rna.csv')
        tail_cell_gene_rna_df = tail_cell_gene_rna_df.replace(['missing'], 0.0)
        tail_cell_gene_rna_df.to_csv('../' + dataset +'/filtered_data/final_rna.csv', index=False, header=True)
        # [CNV]
        tail_cell_gene_cnv_df = pd.read_csv('../' + dataset +'/mid_gene/tail_cell_gene_cnv.csv')
        tail_cell_gene_cnv_df = tail_cell_gene_cnv_df.replace(['missing'], 0.0)
        tail_cell_gene_cnv_df.to_csv('../' + dataset +'/filtered_data/final_cnv.csv', index=False, header=True)
        # [Methylation]
        tail_cell_gene_max_cmeth_df = pd.read_csv('../' + dataset +'/mid_gene/tail_cell_gene_cmeth_max.csv')
        tail_cell_gene_max_cmeth_df.to_csv('../' + dataset +'/filtered_data/final_cmeth_max.csv', index=False, header=True)
        tail_cell_gene_min_cmeth_df = pd.read_csv('../' + dataset +'/mid_gene/tail_cell_gene_cmeth_min.csv')
        tail_cell_gene_min_cmeth_df.to_csv('../' + dataset +'/filtered_data/final_cmeth_min.csv', index=False, header=True)
        # print(tail_cell_gene_cmeth_df.isnull().values.any())
        # [CCLE AMP]
        tail_cell_gene_cmut_amp_df = pd.read_csv('../' + dataset +'/mid_gene/tail_cell_gene_cmut_amp.csv')
        # print(tail_cell_gene_cmut_amp_df.isna().sum().sum())
        tail_cell_gene_cmut_amp_df.to_csv('../' + dataset +'/filtered_data/final_cmut_amp.csv', index=False, header=True)
        # [CCLE DEL]
        tail_cell_gene_cmut_del_df = pd.read_csv('../' + dataset +'/mid_gene/tail_cell_gene_cmut_del.csv')
        # print(tail_cell_gene_cmut_del_df.isna().sum().sum())
        tail_cell_gene_cmut_del_df.to_csv('../' + dataset +'/filtered_data/final_cmut_del.csv', index=False, header=True)
        # [DrugBank]
        tail_gene_drug_drugbank_df = pd.read_csv('../' + dataset +'/mid_drug/tail_gene_drug_drugbank.csv')
        tail_gene_drug_drugbank_df = tail_gene_drug_drugbank_df.sort_values(by=['Drug', 'Target'])
        tail_gene_drug_drugbank_df.to_csv('../' + dataset +'/filtered_data/final_drugbank.csv', index=False, header=True)
        tail_gene_drug_drugbank_genelist = set(list(tail_gene_drug_drugbank_df['Target']))
        print('----- NUMBER OF GENEs IN DRUGBANK: ' + str(len(tail_gene_drug_drugbank_genelist)) + ' ------')
        print('----- NUMBER OF DRUGs IN DRUGBANK: ' + str(len(set(list(tail_gene_drug_drugbank_df['Drug'])))) + ' ------')
        print(sorted(list(set(tail_gene_drug_drugbank_df['Drug']))))


### DATASET SELECTION
# dataset = 'datainfo-nci'
dataset = 'datainfo-oneil'

if os.path.exists('../' + dataset +'/mid_gene') == False:
    os.mkdir('../' + dataset +'/mid_gene')
if os.path.exists('../' + dataset +'/mid_cell_line') == False:
    os.mkdir('../' + dataset +'/mid_cell_line')
if os.path.exists('../' + dataset +'/mid_drug') == False:
    os.mkdir('../' + dataset +'/mid_drug')
if os.path.exists('../' + dataset +'/filtered_data') == False:
    os.mkdir('../' + dataset +'/filtered_data')

# CellLineAnnotation().parse_cell_xml(dataset)
CellLineAnnotation().dl_cell_annotation(dataset)
CellLineAnnotation().omics_cell(dataset)
CellLineAnnotation().tail_cell(dataset)

GeneAnnotation().check_gdsc_cnv_tail_overzero(dataset)
GeneAnnotation().check_gdsc_rna_tail_overzero(dataset)
GeneAnnotation().kegg_omics_intersect(dataset)
GeneAnnotation().kegg_drugbank_gene_intersect(dataset)
GeneAnnotation().kegg_path_omics_intersect(dataset)

DrugAnnotation().nci_drugbank_drug_intersect(dataset)

RecheckFinal().recheck_cell_line(dataset)
RecheckFinal().final(dataset)