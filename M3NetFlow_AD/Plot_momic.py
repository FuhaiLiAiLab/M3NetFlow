import os
import shutil
import pandas as pd
import networkx as nx
from scipy.stats import chi2_contingency
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

    
    def filter_edges(patient_type):

        df = pd.read_csv(f'./ROSMAP-analysis/avg_analysis/average_attention_{patient_type}.csv')
        df.to_csv(f'./ROSMAP-analysis/avg_analysis/filtered_average_attention_{patient_type}.csv', index=False)
        
    filter_edges(patient_type_one)
    filter_edges(patient_type_two)

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
    if patient_type == 'AD':
        map_all_gene_weight_df.to_csv('./ROSMAP-analysis/avg_analysis/map-all-gene-AD-att_deg.csv', index=False)
    elif patient_type == 'NOAD':
        map_all_gene_weight_df.to_csv('./ROSMAP-analysis/avg_analysis/map-all-gene-NOAD-att_deg.csv', index=False)


###############Main workflow###############

# Define patient types

patient_type_one = 'AD'
patient_type_two = 'NOAD'

threshold = 0.1
top = 50
giant_comp_threshold = 20

# 1. Split Files Based on AD and calculate average attention
split_files_and_calculate_average_attention(patient_type_one, patient_type_two)

# t=0.1  p<0.2 | 0.1 ratio  (base 400 genes 120<0.2 pvalues  120/400=0.3 ratio) gene node weighted degree
# t=0.15 p<0.2 | 0.1 ratio  80
# t=0.2  p<0.2 | 0.1 ratio  increases 70
# t=0.3  p<0.2 | 0.1 ratio  converges 50