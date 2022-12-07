from __future__ import print_function
import os
import json
import random
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
from submodlib.helper import create_kernel
from submodlib.functions.facilityLocationMutualInformation import FacilityLocationMutualInformationFunction
from submodlib.functions.facilityLocationVariantMutualInformation import FacilityLocationVariantMutualInformationFunction
from submodlib.functions.graphCutMutualInformation import GraphCutMutualInformationFunction
from submodlib.functions.logDeterminantMutualInformation import LogDeterminantMutualInformationFunction

random.seed(42)

def budget(num_samples):
    return int(3.6*num_samples)

accent_map = {"ABA":"arabic","SKA":"arabic","YBAA":"arabic","ZHAA":"arabic",
              "BWC":"chinese","LXC":"chinese","NCC":"chinese","TXHC":"chinese",
              "ASI":"hindi","RRBI":"hindi","SVBI":"hindi","TNI":"hindi",
              "HJK":"korean","HKK":"korean","YDCK":"korean","YKWK":"korean",
              "EBVS":"spanish","ERMS":"spanish","MBMPS":"spanish","NJS":"spanish",
              "HQTV":"vietnamese","PNV":"vietnamese","THV":"vietnamese","TLV":"vietnamese"
              }
accent_short_forms = {"hindi":"HIN", "korean":"KOR", "vietnamese":"VTN", "arabic":"ARB", "chinese":"CHN", "spanish":"ESP"}
composed_accent_map = {k: accent_short_forms.get(v) for k, v in accent_map.items()}

accent_list = [k for k,v in accent_short_forms.items()]

def plots(dirs, run_dir, ground_features, ground_features_Y, query_features, test_features, selected_indices, selected_gains, fxn):
    # plot_TSNE(dirs, run_dir, ground_features, ground_features_Y, query_features, test_features, selected_indices, selected_gains, fxn)  
    # plot_PCA(dirs, run_dir, ground_features, ground_features_Y, query_features, test_features, selected_indices, selected_gains, fxn)
    pass

def compute_subset(dirs, base_dir, query_dir, ground_list, ground_features, ground_features_Y, query_list, query_features, test_features, greedyList, budget_size, target_size, speaker, fxn, similarity, etaValue, feature_type):
    list_total_selection, list_total_count, list_total_duration = [], [], []
    list_speaker_sample_count, list_speaker_sample_duration = [], []
    accent_selection = []
    output_dir = os.path.join(base_dir, query_dir, f'TSS_output/all/budget_{budget_size}/target_{target_size}/{fxn}/eta_{etaValue}/{similarity}/{feature_type}')
    os.makedirs(output_dir, exist_ok=True)
    for i in [1, 2, 3]:
        run = f'run_{i}'
        run_dir = os.path.join(output_dir, run)
#         for folder in ['train', 'output', 'plots']:
#             os.makedirs(os.path.join(run_dir, folder), exist_ok=True)
            
        all_indices = [j[0] for j in greedyList]
        all_gains = [j[1] for j in greedyList]
        total_duration, index = 0, 0
        while total_duration + ground_list[all_indices[index]]['duration'] <= budget(budget_size):
            total_duration += ground_list[all_indices[index]]['duration']
            index += 1

        list_total_count.append(index)
        list_total_duration.append(total_duration)
        selected_indices = all_indices[:index]
        selected_gains = all_gains[:index]
        selected_list = [ground_list[j] for j in selected_indices]
        
        speaker_sample_count, speaker_sample_duration = 0, 0
        for item in selected_list:
            if accent_map[item['audio_filepath'].split('/')[-3]] == accent_map[speaker]:
                speaker_sample_count += 1
                speaker_sample_duration += item['duration']
        list_speaker_sample_count.append(speaker_sample_count)
        list_speaker_sample_duration.append(speaker_sample_duration)
        list_total_selection.append(Counter([item['audio_filepath'].split('/')[-3] for item in selected_list]))
        accent_selection.append(Counter([accent_map[item['audio_filepath'].split('/')[-3]] for item in selected_list]))
#        train_list = selected_list + query_list
        train_list = selected_list

        with open('./selections/{}-{}-{}-{}.json'.format(speaker, fxn, budget_size, target_size), 'w') as f:
            for line in train_list:
                f.write('{}\n'.format(json.dumps(line)))

#        plots(dirs, run_dir, ground_features, ground_features_Y, query_features, test_features, selected_indices, selected_gains, fxn)

    print('subset computed ....')
    stats = 'total selection : ' + ' '.join(map(str, list_total_count)) + ' -> {0:.2f}\n'.format(statistics.mean(list_total_count))
    stats += 'total selection duration: ' + ' '.join(map(str, list_total_duration)) + ' -> {0:.2f}\n'.format(statistics.mean(list_total_duration))
    stats += 'speaker selection: ' + ' '.join(map(str, list_speaker_sample_count)) + ' -> {0:.2f}\n'.format(statistics.mean(list_speaker_sample_count))
    stats += 'speaker duration: ' + ' '.join(map(str, list_speaker_sample_duration)) + ' -> {0:.2f}\n'.format(statistics.mean(list_speaker_sample_duration))
    stats += '\naccents: ' + str(accent_selection)
    stats += '\nall selections: ' + str(list_total_selection)
    
#     with open(output_dir + '/stats.txt', 'w') as f:
#         f.write(stats)
    print(stats)

def generate_greedyList(ground_kernel, query_kernel, query_query_kernel, fxn, budget_size, etaValue):
    print(f'\ncreating {fxn} object\n')
    if fxn == 'FL1MI':
        obj1 = FacilityLocationMutualInformationFunction(n=len(ground_kernel), num_queries=query_kernel.shape[1], query_sijs=query_kernel, data_sijs=ground_kernel, magnificationEta=etaValue)
    elif fxn == 'FL2MI':
        obj1 = FacilityLocationVariantMutualInformationFunction(n=len(ground_kernel), num_queries=query_kernel.shape[1], query_sijs=query_kernel, queryDiversityEta=etaValue)
    elif fxn == 'GCMI':
        obj1 = GraphCutMutualInformationFunction(n=len(ground_kernel), num_queries=query_kernel.shape[1], query_sijs=query_kernel)
    elif fxn == 'LogDMI':
        obj1 = LogDeterminantMutualInformationFunction(n=len(ground_kernel), num_queries=query_kernel.shape[1], lambdaVal=1, query_sijs=query_kernel, data_sijs=ground_kernel, query_query_sijs=query_query_kernel, magnificationEta=etaValue)
    else:
        print('............... ERROR not a valid FUNCTION ............')
        exit()
    print('{} object created, generating greedy list'.format(fxn))
    greedyList = obj1.maximize(budget=3*budget_size, optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, epsilon=0.1, verbose=False)
    print('generated')
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

def preprocess(base_dir, target_size, budget_size, speaker, similarity, feature_type):
    dirs = ['ABA', 'SKA', 'YBAA', 'ZHAA', 'BWC', 'LXC', 'NCC', 'TXHC', 'ASI', 'RRBI', 'SVBI', 'TNI', 'HJK', 'HKK', 'YDCK', 'YKWK', 'EBVS', 'ERMS', 'MBMPS', 'NJS', 'HQTV', 'PNV', 'THV', 'TLV']
    dirs.remove(speaker)
    print('__', dirs)

    query_dir = f'{speaker}/manifests/' 
    query_file_path = base_dir + query_dir + 'seed.json'
    query_list = [json.loads(line.strip()) for line in open(query_file_path)]
    query_features = load_features(query_file_path, feature_type)
    query_list, query_features = query_list[:target_size], query_features[:target_size]

    ground_list, ground_list_Y, ground_features = [], [], []
    for i, _dir in enumerate(dirs):
        if accent_map[_dir] == accent_map[speaker]:
            selection_file_path = base_dir + _dir + '/manifests/selection.json'
            selection_file_list = [json.loads(line.strip()) for line in open(selection_file_path)]
            list_Y = [i]*len(selection_file_list)
            ground_list.extend(selection_file_list[:])
            ground_features.append(load_features(selection_file_path, feature_type))
            ground_list_Y.extend(list_Y)   
        else:
            selection_file_path = base_dir + _dir + '/manifests/selection.json'
            selection_file_list = [json.loads(line.strip()) for line in open(selection_file_path)]
            list_Y = [i]*len(selection_file_list)
            selection_indices = list(range(len(selection_file_list)))
            random.shuffle(selection_indices)
            selection_file_list = [selection_file_list[item] for item in selection_indices[:int(len(selection_indices)*.75)]]
            ground_list.extend(selection_file_list[:])
            ground_features.append(load_features(selection_file_path, feature_type)[selection_indices[:int(len(selection_indices)*.75)]])
            list_Y = [list_Y[item] for item in selection_indices[:int(len(selection_indices)*.75)]]
            ground_list_Y.extend(list_Y)   
    ground_features = np.concatenate(ground_features, axis=0)
    ground_features_Y = np.asarray(ground_list_Y).reshape(-1, 1) 

    ### test file
    test_file_path = base_dir + query_dir + 'test.json'
    test_list = [json.loads(line.strip()) for line in open(test_file_path)]
    test_features = load_features(test_file_path, feature_type)

    print("{} in ground, shape={}".format(len(ground_list), ground_features.shape))
    print("{} in query, shape={}".format(len(query_list), query_features.shape))
    print("{} in test, shape={}".format(len(test_list), test_features.shape))

    print('creating kernels ....')
    t1 = time.time()
    ground_kernel = create_kernel(ground_features, metric=similarity, mode='dense')
    query_kernel = create_kernel(query_features, metric=similarity, mode='dense', X_rep=ground_features)
    query_query_kernel = create_kernel(query_features, metric=similarity, mode='dense', X_rep=query_features)
    t2 = time.time()
    print("kernel creation done .... in {}s".format(t2-t1))

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
parser.add_argument('--speaker', type = str,  help ='query set')
parser.add_argument('--feature_type', type = str,  help ='which feature space')

args = parser.parse_args()
budget_size = args.budget
target_size = args.target
etaValue = args.eta
similarity = args.similarity
fxn = args.fxn
speaker = args.speaker
feature_type = args.feature_type
base_dir = ''

dirs, query_dir, ground_list, ground_features, ground_features_Y, ground_kernel, query_list, query_features, query_kernel, query_query_kernel, test_features = preprocess(base_dir, target_size, budget_size, speaker, similarity, feature_type)
greedyList = generate_greedyList(ground_kernel, query_kernel, query_query_kernel, fxn, budget_size, etaValue)
compute_subset(dirs, base_dir, query_dir, ground_list, ground_features, ground_features_Y, query_list, query_features, test_features, greedyList, budget_size, target_size, speaker, fxn, similarity, etaValue, feature_type)