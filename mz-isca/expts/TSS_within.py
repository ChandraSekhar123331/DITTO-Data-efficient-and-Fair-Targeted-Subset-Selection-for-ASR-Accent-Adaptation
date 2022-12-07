from __future__ import print_function
import os
import json
import argparse
import statistics
import numpy as np
import pickle
import torch
import random
import pathlib
# from pydub import AudioSegment
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
import submodlib
from submodlib.helper import create_kernel
from submodlib.functions.facilityLocationMutualInformation import FacilityLocationMutualInformationFunction
from submodlib.functions.facilityLocationVariantMutualInformation import FacilityLocationVariantMutualInformationFunction
from submodlib.functions.graphCutMutualInformation import GraphCutMutualInformationFunction
from submodlib.functions.logDeterminantMutualInformation import LogDeterminantMutualInformationFunction


def budget(num_samples):
    return int(540/100*num_samples)

def plots(dirs, run_dir, ground_features, ground_features_Y, query_features, test_features, selected_indices):
    pass

def compute_subset(dirs, base_dir, query_dir, ground_list, budget_size, target_size, accent):
    list_total_selection, list_total_count, list_total_duration = [], [], []
    list_accent_sample_count, list_accent_sample_duration = [], []
    output_dir = os.path.join(base_dir, query_dir, f'TSS_output/within/budget_{budget_size}/target_{target_size}/self')
    os.makedirs(output_dir, exist_ok=True)
    for i in [1, 2, 3]:
        run = f'run_{i}'
        run_dir = os.path.join(output_dir, run)
        for folder in ['train', 'output', 'plots']:
            os.makedirs(os.path.join(run_dir, folder), exist_ok=True)
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
        
        accent_sample_count, accent_sample_duration = 0, 0
        
        for item in selected_list:
            if item['accent'] == accent:
                accent_sample_count += 1
                accent_sample_duration += item['duration']
            if os.path.isfile(item['audio_filepath']):
                pass
            else:
                wav_path = item['audio_filepath']
                mp3name = wav_path.replace('/wav/', '/clips/')
                mp3name = mp3name.replace('.wav', '.mp3')
                sys.exit()
                # sound = AudioSegment.from_mp3(mp3name)
                # sound.export(wav_path, format="wav")
            
        list_accent_sample_count.append(accent_sample_count)
        list_accent_sample_duration.append(accent_sample_duration)
        list_total_selection.append(Counter([item['accent'] for item in selected_list]))
        
#        with open(base_dir + query_dir + f'train/error_model/{budget_size}/seed_{i}/train.json', 'w') as f:
#            for line in train_list:
#                f.write('{}\n'.format(json.dumps(line)))
        print('making', f'{run_dir}/train/train.json')
        with open(f'{run_dir}/train/train.json', 'w') as f:
            for line in train_list:
                f.write('{}\n'.format(json.dumps(line)))
        print("done")
#        plots(dirs, run_dir, ground_features, ground_features_Y, query_features, test_features, selected_indices)
    
    stats = 'total selection : ' + ' '.join(map(str, list_total_count)) + ' -> {0:.2f}\n'.format(statistics.mean(list_total_count))
    stats += 'total selection duration: ' + ' '.join(map(str, list_total_duration)) + ' -> {0:.2f}\n'.format(statistics.mean(list_total_duration))
    stats += 'accented selection: ' + ' '.join(map(str, list_accent_sample_count)) + ' -> {0:.2f}\n'.format(statistics.mean(list_accent_sample_count))
    stats += 'accented duration: ' + ' '.join(map(str, list_accent_sample_duration)) + ' -> {0:.2f}\n'.format(statistics.mean(list_accent_sample_duration))
    stats += '\nall selections: ' + str(list_total_selection)
    
    with open(output_dir + '/stats.txt', 'w') as f:
        f.write(stats)

def load_features(file_dir, feature_type):
    features = []
    with open(file_dir.replace('.json', f'_{feature_type}.file'), 'rb') as f:
        while True:
            try:
                features.append(pickle.load(f))
            except EOFError:
                break
    features = np.concatenate(features, axis=0)
    print(features.shape)
    return features

def preprocess(base_dir, target_size, budget_size, accent, feature_type):
    
    dirs = [accent]
    
    print("target dir", accent)
    print("ground dirs", dirs)
    
    query_dir = f'{accent}/manifests/' 
    query_file_path = base_dir + query_dir + 'seed.json'
    query_list = [json.loads(line.strip()) for line in open(query_file_path)]
    query_features = load_features(query_file_path, feature_type)
    query_list, query_features = query_list[:target_size], query_features[:target_size]

    ground_list, ground_list_Y, ground_features = [], [], []
    test_list, test_features = [], []
    for i, _dir in enumerate(dirs):
        ### ground files
        selection_file_path = base_dir + _dir + '/manifests/selection.json'
        selection_file_list = [json.loads(line.strip()) for line in open(selection_file_path)]
        if selection_file_list==[]: continue
        ground_list.extend(selection_file_list) 

    return dirs, query_dir, ground_list

parser = argparse.ArgumentParser(description ='TSS input')
parser.add_argument('--budget', type = int,  help ='budget')
parser.add_argument('--target', type = int,  help ='target')
parser.add_argument('--accent', type = str,  help ='query set')
parser.add_argument('--feature_type', type = str,  help ='feature space')

args = parser.parse_args()
budget_size = args.budget
target_size = args.target
accent = args.accent
feature_type = args.feature_type
base_dir = ''

dirs, query_dir, ground_list = preprocess(base_dir, target_size, budget_size, accent, feature_type)
compute_subset(dirs, base_dir, query_dir, ground_list, budget_size, target_size, accent)
