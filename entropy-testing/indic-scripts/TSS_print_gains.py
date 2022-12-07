from __future__ import print_function
import os
import json
import argparse
import statistics
import numpy as np
import pickle
import torch
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
import re
from submodlib.helper import create_kernel
from submodlib.functions.facilityLocationMutualInformation import FacilityLocationMutualInformationFunction
from submodlib.functions.facilityLocationVariantMutualInformation import FacilityLocationVariantMutualInformationFunction
from submodlib.functions.graphCutMutualInformation import GraphCutMutualInformationFunction
from submodlib.functions.logDeterminantMutualInformation import LogDeterminantMutualInformationFunction

def budget(budget_size):
    return int(4.92*budget_size)


def compute_subset(dirs, base_dir, query_dir, ground_list, ground_features, ground_features_Y, query_list, query_features, test_features, greedyList, budget_size, target_size, accent, fxn, similarity, etaValue, feature_type):
#     print(greedyList)
    list_total_selection, list_total_count, list_total_duration = [], [], []
    list_accent_sample_count, list_accent_sample_duration = [], []
    output_dir = os.path.join(base_dir, query_dir, f'TSS_output/all/budget_{budget_size}/target_{target_size}/{fxn}/eta_{etaValue}/{similarity}/{feature_type}')
    os.makedirs(output_dir, exist_ok=True)
    for i in [1, 2, 3]:
        run = f'run_{i}'
        run_dir = os.path.join(output_dir, run)
        for folder in ['train', 'output', 'plots']:
            os.makedirs(os.path.join(run_dir, folder), exist_ok=True)
            
        all_indices = [j[0] for j in greedyList]
        all_gains = [j[1] for j in greedyList]
        def get_accent(ind):
            return re.search('/indicTTS/([A-Za-z_\-]+)/', ground_list[ind]['audio_filepath']).group(1)
        all_accents = [get_accent(ind) for ind in all_indices]
#         print(all_accents)
        total_duration, index = 0, 0
        while total_duration + ground_list[all_indices[index]]['duration'] <= budget(budget_size):
            total_duration += ground_list[all_indices[index]]['duration']
            index += 1

        list_total_count.append(index)
        list_total_duration.append(total_duration)
        selected_indices = all_indices[:index]
        selected_gains = all_gains[:index]
        selected_accents = all_accents[:index]
        
        print(Counter(selected_accents))
#         print("Number of repeated values = ", len(selected_indices) - len(set(selected_indices)))
        with open(f'{run_dir}/gains.txt' , 'w') as file:
            for ind, (entry, accent) in enumerate(zip(selected_gains, selected_accents)):
                file.write(f"{ind}, {entry}, {accent}\n")
        excluded = 50
        plt.plot(excluded + np.arange(len(selected_gains[excluded:])), selected_gains[excluded:])
        plt.xlabel("sample number")
        plt.ylabel("selected gains")
        plt.title(f"gains graph excluding first {excluded} samples")
        plt.xscale("log")
        plt.savefig(f"{run_dir}/gains-log_scale_x.png")
        plt.clf()
        
        excluded = 50
        plt.plot(excluded + np.arange(len(selected_gains[excluded:])), selected_gains[excluded:])
        plt.xlabel("sample number")
        plt.ylabel("selected gains")
        plt.title(f"gains graph excluding first {excluded} samples")
        plt.savefig(f"{run_dir}/gains-usual_scale_x.png")
        plt.clf()
                
        selected_list = [ground_list[j] for j in selected_indices]

#        train_list = selected_list + query_list
        train_list = selected_list 
        
        accent_sample_count, accent_sample_duration = 0, 0
        for item in selected_list:
            if item['audio_filepath'].split('/')[-4] == accent:
                accent_sample_count += 1
                accent_sample_duration += item['duration']
        list_accent_sample_count.append(accent_sample_count)
        list_accent_sample_duration.append(accent_sample_duration)
        list_total_selection.append(Counter([item['audio_filepath'].split('/')[-4] for item in selected_list]))
        
#        with open(base_dir + query_dir + f'train/error_model/{budget_size}/seed_{i}/train.json', 'w') as f:
#            for line in train_list:
#                f.write('{}\n'.format(json.dumps(line)))
        with open(f'{run_dir}/train/train.json', 'w') as f:
            for line in train_list:
                f.write('{}\n'.format(json.dumps(line)))

#        plots(dirs, run_dir, ground_features, ground_features_Y, query_features, test_features, selected_indices, selected_gains, fxn)

    print('\n subset computed .... \n')
    stats = 'total selection : ' + ' '.join(map(str, list_total_count)) + ' -> {0:.2f}\n'.format(statistics.mean(list_total_count))
    stats += 'total selection duration: ' + ' '.join(map(str, list_total_duration)) + ' -> {0:.2f}\n'.format(statistics.mean(list_total_duration))
    stats += 'accented selection: ' + ' '.join(map(str, list_accent_sample_count)) + ' -> {0:.2f}\n'.format(statistics.mean(list_accent_sample_count))
    stats += 'accented duration: ' + ' '.join(map(str, list_accent_sample_duration)) + ' -> {0:.2f}\n'.format(statistics.mean(list_accent_sample_duration))
    stats += '\nall selections: ' + str(list_total_selection)
    
    with open(output_dir + '/stats.txt', 'w') as f:
        f.write(stats)

def generate_greedyList(ground_kernel, query_kernel, query_query_kernel, fxn, budget_size, etaValue):
    print(f'\ncreating {fxn} object\n')
    if fxn == 'FL1MI':
        obj1 = FacilityLocationMutualInformationFunction(n=len(ground_kernel), num_queries=query_kernel.shape[1],
                                                         query_sijs=query_kernel, data_sijs=ground_kernel, 
                                                         magnificationEta=etaValue)
    elif fxn == 'FL2MI':
        obj1 = FacilityLocationVariantMutualInformationFunction(n=len(ground_kernel), num_queries=query_kernel.shape[1],
                                                         query_sijs=query_kernel, queryDiversityEta=etaValue)
    elif fxn == 'GCMI':
        obj1 = GraphCutMutualInformationFunction(n=len(ground_kernel), num_queries=query_kernel.shape[1],
                                                         query_sijs=query_kernel)
    elif fxn == 'LogDMI':
        obj1 = LogDeterminantMutualInformationFunction(n=len(ground_kernel), num_queries=query_kernel.shape[1], lambdaVal=1,
                                                         query_sijs=query_kernel, data_sijs=ground_kernel, 
                                                         query_query_sijs=query_query_kernel, magnificationEta=etaValue)
    else:
        print('\n\n\n............... ERROR not a valid FUNCTION ............\n\n\n')
        exit()
    print(f'\n{fxn} object created\n')
    print('\ngenerating greedyList...\n') 
    greedyList = obj1.maximize(budget=3*budget_size, optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, epsilon=0.1, verbose=False)
    print('\n.... greedyList generated ... \n')
    return greedyList

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

def preprocess(base_dir, target_size, budget_size, accent, similarity, feature_type):
    dirs = ['kannada_male_english', 'malayalam_male_english', 'rajasthani_male_english', 'hindi_male_english', 'tamil_male_english', 'gujarati_female_english', 'manipuri_female_english', 'assamese_female_english']
#     dirs = ['kannada_male_english', 'manipuri_female_english', 'assamese_female_english']
    query_dir = f'{accent}/manifests/' 
    query_file_path = base_dir + query_dir + 'seed.json'
    query_list = [json.loads(line.strip()) for line in open(query_file_path)]
    query_features = load_features(query_file_path, feature_type)
    query_list, query_features = query_list[:target_size], query_features[:target_size]

    ground_list, ground_list_Y, ground_features = [], [], []
    for i, _dir in enumerate(dirs):
        selection_file_path = base_dir + _dir + '/manifests/selection.json'
        selection_file_list = [json.loads(line.strip()) for line in open(selection_file_path)]
        ground_list.extend(selection_file_list)
        ground_features.append(load_features(selection_file_path, feature_type))
        ground_list_Y.extend([i]*len(selection_file_list))   
    ground_features = np.concatenate(ground_features, axis=0)
    ground_features_Y = np.asarray(ground_list_Y).reshape(-1, 1) 

    ### test file
    test_file_path = base_dir + query_dir + 'test.json'
    test_list = [json.loads(line.strip()) for line in open(test_file_path)]
    test_features = load_features(test_file_path, feature_type)

    print(len(ground_list), ground_features.shape)
    print(len(query_list), query_features.shape)
    print(len(test_list), test_features.shape)

    print('creating kernels ....')
    t1 = time.time()
    ground_kernel = create_kernel(ground_features, metric=similarity, mode='dense')
    query_kernel = create_kernel(query_features, metric=similarity, mode='dense', X_rep=ground_features)
    query_query_kernel = create_kernel(query_features, metric=similarity, mode='dense', X_rep=query_features)
    t2 = time.time()
    print('kernel creation done ....', t2-t1)

    print('ground_kernel: ', ground_kernel.shape)
    print('query_kernel: ', query_kernel.shape)
    print('query_query_kernel: ', query_query_kernel.shape)

    return dirs, query_dir, ground_list, ground_features, ground_features_Y, ground_kernel, query_list, query_features, query_kernel, query_query_kernel, test_features 

parser = argparse.ArgumentParser(description ='TSS input')
parser.add_argument('--budget', type = int,  help ='budget')
parser.add_argument('--target', type = int,  help ='target')
parser.add_argument('--eta', type = float,  help ='eta value')
parser.add_argument('--similarity', type = str,  help ='similarity metric')
parser.add_argument('--fxn', type = str,  help ='function')
parser.add_argument('--accent', type = str,  help ='query set')
parser.add_argument('--feature_type', type = str,  help ='which feature space')

args = parser.parse_args()
budget_size = args.budget
target_size = args.target
etaValue = args.eta
similarity = args.similarity
fxn = args.fxn
accent = args.accent
feature_type = args.feature_type
base_dir = ''

dirs, query_dir, ground_list, ground_features, ground_features_Y, ground_kernel, query_list, query_features, query_kernel, query_query_kernel, test_features = preprocess(base_dir, target_size, budget_size, accent, similarity, feature_type)
greedyList = generate_greedyList(ground_kernel, query_kernel, query_query_kernel, fxn, budget_size, etaValue)
compute_subset(dirs, base_dir, query_dir, ground_list, ground_features, ground_features_Y, query_list, query_features, test_features, greedyList, budget_size, target_size, accent, fxn, similarity, etaValue, feature_type)
