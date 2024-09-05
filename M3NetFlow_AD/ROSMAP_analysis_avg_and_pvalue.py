import os
import shutil
import pandas as pd
import networkx as nx
from scipy.stats import chi2_contingency
from scipy.stats import mannwhitneyu
import re
import subprocess
import matplotlib.pyplot as plt
import numpy as np

# 1. Split Files Based on AD and calculate average attention
def split_files_and_calculate_average_attention(patient_type_one, patient_type_two):
    map_dict_path = 'ROSMAP-graph-data/survival_label_map_dict.csv'
    map_dict_df = pd.read_csv(map_dict_path)
    num_to_id_dict = pd.Series(map_dict_df['individualID'].values, index=map_dict_df['individualID_Num']).to_dict()

    label_path = 'ROSMAP-graph-data/random-survival-label.csv'
    label_df = pd.read_csv(label_path)
    if patient_type_one in ['AD', 'NOAD']:
        id_to_dict = pd.Series(label_df['ceradsc'].values, index=label_df['individualID']).to_dict()
    else:
        id_to_dict = pd.Series(label_df['msex'].values, index=label_df['individualID']).to_dict()

    survival_dir = './ROSMAP-analysis/avg/'
    files = os.listdir(survival_dir)
    os.makedirs('./ROSMAP-analysis/avg_analysis', exist_ok=True)
    patient_type_one_dir = f'./ROSMAP-analysis/avg_analysis/{patient_type_one}'
    patient_type_two_dir = f'./ROSMAP-analysis/avg_analysis/{patient_type_two}'

    # Make directories if they don't exist
    os.makedirs(patient_type_one_dir, exist_ok=True)
    os.makedirs(patient_type_two_dir, exist_ok=True)

    for file in files:
        if file.endswith('.csv'):
            num = int(file.split('survival')[1].split('.csv')[0])

            if num in num_to_id_dict:
                individual_id = num_to_id_dict[num]

                if individual_id in id_to_dict:
                    value = id_to_dict[individual_id]

                    if value == 0:
                        shutil.copy(os.path.join(survival_dir, file), os.path.join(patient_type_two_dir, file))
                    elif value == 1:
                        shutil.copy(os.path.join(survival_dir, file), os.path.join(patient_type_one_dir, file))

    def calculate_average_attention(folder_path):
        all_data = []
        key_columns = ['From', 'To', 'Hop', 'SignalingPath', 'SpNotation']
        # Read each file and collect the data
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path)
                # Select relevant columns, ensuring 'individualID' is not included
                if 'individualID' in df.columns:
                    df = df.drop(columns=['individualID'])
                # Check if all necessary columns are present
                if all(col in df.columns for col in key_columns + ['Attention']):
                    all_data.append(df)
                else:
                    print(f"File {filename} is missing one of the required columns.")
        # Concatenate all the dataframes in the list
        if not all_data:
            print("No valid files to process.")
            return None
        combined_df = pd.concat(all_data, ignore_index=True)
        # Group by the key columns and calculate the mean of 'Attention'
        result_df = combined_df.groupby(key_columns)['Attention'].mean().reset_index()
        return result_df

    patient_type_one_result_df = calculate_average_attention(patient_type_one_dir)
    patient_type_one_result_df.to_csv(f'./ROSMAP-analysis/avg_analysis/average_attention_{patient_type_one}.csv', index=False)
    patient_type_two_result_df = calculate_average_attention(patient_type_two_dir)
    patient_type_two_result_df.to_csv(f'./ROSMAP-analysis/avg_analysis/average_attention_{patient_type_two}.csv', index=False)
    patient_type_one_name_result_df = add_node_name(patient_type_one_result_df)
    patient_type_two_name_result_df = add_node_name(patient_type_two_result_df)
    patient_type_one_name_result_df.to_csv(f'./ROSMAP-analysis/avg_analysis/average_attention_{patient_type_one}_name.csv', index=False)
    patient_type_two_name_result_df.to_csv(f'./ROSMAP-analysis/avg_analysis/average_attention_{patient_type_two}_name.csv', index=False)
    calculate_weighted_degree(patient_type_one_name_result_df, patient_type_one)
    calculate_weighted_degree(patient_type_two_name_result_df, patient_type_two)

def add_node_name(df):
    map_all_gene_df = pd.read_csv('./ROSMAP-graph-data/map-all-gene.csv')
    map_all_gene_dict = dict(zip(map_all_gene_df['Gene_num'], map_all_gene_df['Gene_name']))
    df['From'] = df['From'].replace(map_all_gene_dict)
    df['To'] = df['To'].replace(map_all_gene_dict)
    return df

def calculate_weighted_degree(df, patient_type):
    # Calculate the weighted degree for each node
    weighted_degree_from = df.groupby('From')['Attention'].sum().reset_index()
    weighted_degree_to = df.groupby('To')['Attention'].sum().reset_index()

    print(weighted_degree_from)
    print(weighted_degree_to)

    # import pdb; pdb.set_trace()
    from_attention = np.array(weighted_degree_from['Attention'].tolist())
    to_attention = np.array(weighted_degree_to['Attention'].tolist())
    average_attention_list = ((from_attention + to_attention)/2).tolist()

    map_all_gene_df = pd.read_csv('./ROSMAP-graph-data/map-all-gene.csv')
    map_all_gene_weight_df = map_all_gene_df.copy()
    map_all_gene_weight_df['Att_deg'] = average_attention_list
    map_all_gene_weight_df.to_csv('./ROSMAP-analysis/avg_analysis/map-all-gene-' + patient_type + '-att_deg.csv', index=False)


# 2. Calculate pvalue for all genes
def calculate_pvalue(patient_type_one, patient_type_two):
   # count survival numbers for each patient type
    patient_type_one_dir = f'./ROSMAP-analysis/avg_analysis/{patient_type_one}'
    patient_type_two_dir = f'./ROSMAP-analysis/avg_analysis/{patient_type_two}'

    def get_survival_numbers(folder_path):
        files = os.listdir(folder_path)
        survival_numbers = []
        for file in files:
            match = re.match(r'survival(\d+)\.csv', file)
            if match:
                number = int(match.group(1))
                survival_numbers.append(number)
        return survival_numbers

    survival_numbers_patient_type_one = sorted(get_survival_numbers(patient_type_one_dir))
    survival_numbers_patient_type_two = sorted(get_survival_numbers(patient_type_two_dir))
    df = pd.read_csv('./ROSMAP-graph-data/survival_label_map_dict.csv')

    def num_to_id(num):
        return df.loc[df['individualID_Num'] == num, 'individualID'].values[0]


    survival_numbers_patient_type_one = [num_to_id(num) for num in survival_numbers_patient_type_one]
    survival_numbers_patient_type_two = [num_to_id(num) for num in survival_numbers_patient_type_two]
    print("Survival numbers:", survival_numbers_patient_type_one)
    print("Survival numbers:", survival_numbers_patient_type_two)

    # count genes for each patient type
    gene_names_df = pd.read_csv(f'./ROSMAP-graph-data/map-all-gene.csv')
    gene_names= gene_names_df['Gene_name'].tolist()
    # gene_names = [name.replace('-PROT', '') for name in gene_names]
    print(gene_names)

    files_to_process = [
        './ROSMAP-process/processed-genotype-cnv_del.csv',
        './ROSMAP-process/processed-genotype-cnv_dup.csv',
        './ROSMAP-process/processed-genotype-cnv_mcnv.csv',
        './ROSMAP-process/processed-genotype-gene-expression.csv',
        './ROSMAP-process/processed-genotype-methy-Core-Promoter.csv',
        './ROSMAP-process/processed-genotype-methy-Distal-Promoter.csv',
        './ROSMAP-process/processed-genotype-methy-Downstream.csv',
        './ROSMAP-process/processed-genotype-methy-Proximal-Promoter.csv',
        './ROSMAP-process/processed-genotype-methy-Upstream.csv',
        './ROSMAP-process/processed-genotype-proteomics.csv',
    ]

    output_dir_patient_type_one = f'./ROSMAP-analysis/node_analysis_processed/{patient_type_one}/'
    output_dir_patient_type_two = f'./ROSMAP-analysis/node_analysis_processed/{patient_type_two}/'
    # mkdir
    os.makedirs(output_dir_patient_type_one, exist_ok=True)
    os.makedirs(output_dir_patient_type_two, exist_ok=True)

    #split files for each patient type
    def process_data(file_path, gene_names, survival_numbers, output_path):
        data = pd.read_csv(file_path)

        columns_to_keep = [num for num in survival_numbers]
        columns_to_keep = ['gene_name'] + columns_to_keep 
        filtered_data = data[columns_to_keep]

        filtered_data = filtered_data[filtered_data['gene_name'].isin(gene_names)]

        filtered_data.to_csv(output_path, index=False)
        print(f"Filtered data saved to {output_path}")

    for file_path in files_to_process:
        file_name = file_path.split('/')[-1].replace('.csv', f'_{patient_type_one}.csv')
        output_path = f"{output_dir_patient_type_one}{file_name}"
        process_data(file_path, gene_names, survival_numbers_patient_type_one, output_path)

    for file_path in files_to_process:
        file_name = file_path.split('/')[-1].replace('.csv', f'_{patient_type_two}.csv')
        output_path = f"{output_dir_patient_type_two}{file_name}"
        process_data(file_path, gene_names, survival_numbers_patient_type_two, output_path)
    

    patient_type_one_files = [
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_one}/processed-genotype-cnv_del_{patient_type_one}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_one}/processed-genotype-cnv_dup_{patient_type_one}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_one}/processed-genotype-cnv_mcnv_{patient_type_one}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_one}/processed-genotype-gene-expression_{patient_type_one}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_one}/processed-genotype-methy-Core-Promoter_{patient_type_one}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_one}/processed-genotype-methy-Distal-Promoter_{patient_type_one}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_one}/processed-genotype-methy-Downstream_{patient_type_one}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_one}/processed-genotype-methy-Proximal-Promoter_{patient_type_one}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_one}/processed-genotype-methy-Upstream_{patient_type_one}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_one}/processed-genotype-proteomics_{patient_type_one}.csv',
    ]

    patient_type_two_files = [
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_two}/processed-genotype-cnv_del_{patient_type_two}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_two}/processed-genotype-cnv_dup_{patient_type_two}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_two}/processed-genotype-cnv_mcnv_{patient_type_two}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_two}/processed-genotype-gene-expression_{patient_type_two}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_two}/processed-genotype-methy-Core-Promoter_{patient_type_two}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_two}/processed-genotype-methy-Distal-Promoter_{patient_type_two}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_two}/processed-genotype-methy-Downstream_{patient_type_two}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_two}/processed-genotype-methy-Proximal-Promoter_{patient_type_two}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_two}/processed-genotype-methy-Upstream_{patient_type_two}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_two}/processed-genotype-proteomics_{patient_type_two}.csv',
    ]

    # mkdir pvalues folder
    os.makedirs('./ROSMAP-analysis/node_analysis_processed/pvalues/', exist_ok=True)

    def calculate(AD_file_path, NOAD_file_path):
        AD_data = pd.read_csv(AD_file_path)
        NOAD_data = pd.read_csv(NOAD_file_path)

        results = []

        for gene in AD_data['gene_name']:
            AD_row = AD_data[AD_data['gene_name'] == gene].iloc[:, 1:].values.flatten()
            NOAD_row = NOAD_data[NOAD_data['gene_name'] == gene].iloc[:, 1:].values.flatten()
            # for gene only in ad or noad
            if len(AD_row) == 0 or len(NOAD_row) == 0:
                print(f"Skipping gene {gene}: One of the rows is empty")
                continue

            stat, p = mannwhitneyu(AD_row, NOAD_row)
            results.append({'gene_name': gene, 'p_value': p})

        results_df = pd.DataFrame(results)
        return results_df

    for patient_type_one_file, patient_type_two_file in zip(patient_type_one_files, patient_type_two_files):
        results_df = calculate(patient_type_one_file, patient_type_two_file)
        file_name = patient_type_one_file.split('/')[-1].replace(f'_{patient_type_one}.csv', '_pvalues.csv')
        results_df.to_csv(f'./ROSMAP-analysis/node_analysis_processed/pvalues/{file_name}', index=False)
        print(f"P-values calculated and saved for {file_name}")



if __name__ == '__main__':
    ###############Main workflow###############
    # Define patient types
    patient_type_one = 'AD'
    patient_type_two = 'NOAD'

    # 1. Split Files Based on AD and calculate average attention
    split_files_and_calculate_average_attention(patient_type_one, patient_type_two)

    # 2. Calculate pvalue for each gene_name for 10 different data types
    calculate_pvalue(patient_type_one, patient_type_two)