from __future__ import print_function
import os
import json
import argparse
import statistics
import numpy as np
import pickle
import torch
import random
from collections import Counter
import time
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def budget(budget_size):
    return int(4.92*budget_size)



def select(base_dir, budget_size):
    accents = ['kannada_male_english', 'malayalam_male_english', 'rajasthani_male_english', 'hindi_male_english', 'tamil_male_english', 'gujarati_female_english', 'manipuri_female_english', 'assamese_female_english']
    
    json_file = os.path.join(base_dir, 'train.json')

    ground_list, ground_list_Y, ground_features = [], [], []
    selection_file_list = [json.loads(line.strip()) for line in open(json_file)]
    ground_list.extend(selection_file_list)
    ground_list_Y.extend([0]*len(selection_file_list))   
    ground_features_Y = np.asarray(ground_list_Y).reshape(-1, 1) 

    print("ground len:{}".format(len(ground_list)))

    output_dir = 'budget_{}/random/'.format(budget_size)
    output_dir = os.path.join(base_dir, output_dir)
    
    list_total_selection, list_total_count, list_total_duration = [], [], []
    list_accent_sample_count, list_accent_sample_duration = {}, {}
    for accent in accents:
        list_accent_sample_count[accent] = []
        list_accent_sample_duration[accent] = []
    
    for i in [1, 2, 3]:
        random.seed(i)
        run = f'run_{i}'
        run_dir = os.path.join(output_dir, run)
        os.makedirs(run_dir, exist_ok=True)
        # for folder in ['train', 'output', 'plots']:
        #     os.makedirs(os.path.join(run_dir, folder))
        all_indices = list(range(len(ground_list)))
        random.seed(i)
        random.shuffle(all_indices)
        total_duration, index = 0, 0
        while total_duration + ground_list[all_indices[index]]['duration'] <= budget(budget_size):
            total_duration += ground_list[all_indices[index]]['duration']
            index += 1
        list_total_count.append(index)
        list_total_duration.append(total_duration)
        selected_indices = all_indices[:index]
        selected_list = [ground_list[j] for j in selected_indices]

#        train_list = selected_list + query_list
        train_list = selected_list
        
        for accent in accents:
            accent_sample_count, accent_sample_duration = 0, 0
            for item in selected_list:
                if item['audio_filepath'].split('/')[-4] == accent:
                    accent_sample_count += 1
                    accent_sample_duration += item['duration']
            list_accent_sample_count[accent].append(accent_sample_count)
            list_accent_sample_duration[accent].append(accent_sample_duration)
        list_total_selection.append(Counter([item['audio_filepath'].split('/')[-4] for item in selected_list]))
        
#        with open(base_dir + query_dir + f'train/error_model/{budget_size}/seed_{i}/train.json', 'w') as f:
#            for line in train_list:
#                f.write('{}\n'.format(json.dumps(line)))
        with open(f'{run_dir}/train.json', 'w') as f:
            for line in train_list:
                f.write('{}\n'.format(json.dumps(line)))
                
#        plots(dirs, run_dir, ground_features, ground_features_Y, query_features, test_features, selected_indices)
    
    stats = 'total selection : ' + ' '.join(map(str, list_total_count)) + ' -> {0:.2f}\n'.format(statistics.mean(list_total_count))
    stats += 'total selection duration: ' + ' '.join(map(str, list_total_duration)) + ' -> {0:.2f}\n'.format(statistics.mean(list_total_duration))
    for accent in accents:
        stats += '\naccent: {}\n'.format(accent)
        stats += 'accented selection: ' + ' '.join(map(str, list_accent_sample_count[accent])) + ' -> {0:.2f}\n'.format(statistics.mean(list_accent_sample_count[accent]))
        stats += 'accented duration: ' + ' '.join(map(str, list_accent_sample_duration[accent])) + ' -> {0:.2f}\n'.format(statistics.mean(list_accent_sample_duration[accent]))
    stats += '\nall selections: ' + str(list_total_selection)
    
    with open(os.path.join(output_dir, 'stats.txt'), 'w') as f:
        f.write(stats)

parser = argparse.ArgumentParser(description ='TSS input')
parser.add_argument('--budget', type = int,  help ='budget')
parser.add_argument('--file_dir', type = str,  help ='dir where json file is')

args = parser.parse_args()
budget_size = args.budget
base_dir = args.file_dir

select(base_dir, budget_size)
