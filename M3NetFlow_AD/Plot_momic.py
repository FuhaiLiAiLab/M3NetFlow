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

    label_path = 'ROSMAP-graph-data/survival-label.csv'
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

    
    def filter_edges(patient_type):

        df = pd.read_csv(f'./ROSMAP-analysis/avg_analysis/average_attention_{patient_type}.csv')
        df.to_csv(f'./ROSMAP-analysis/avg_analysis/filtered_average_attention_{patient_type}.csv', index=False)
        
    filter_edges(patient_type_one)
    filter_edges(patient_type_two)



###############Main workflow###############

# Define patient types

patient_type_one = 'AD'
patient_type_two = 'NOAD'

# 1. Split Files Based on AD and calculate average attention
split_files_and_calculate_average_attention(patient_type_one, patient_type_two)

# 2. Plot the graph