#%%
import os
import json 
import random
from tqdm import tqdm
import pandas as pd
import librosa
from pathlib import Path

#%%

#%%
def create_file_MI(accent, input_file, fxn):
    data = []
    home_path = '/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/mz-isca/expts/'
    #json_path = os.path.join(home_path, accent, f'manifests/{input_file}')
    json_path = os.path.join(home_path, accent, f'manifests/TSS_output/all/budget_200/target_20/{fxn}/eta_1.0/euclidean/wv10_100/run_1/train/')
    json_file = open(json_path+input_file)
    json_item_list = [line for line in json_file]
    output_file = input_file.replace('.json', f'.csv')
    json_item_list = [json.loads(line.strip()) for line in json_item_list]
    for sample in tqdm(json_item_list):
     try:
         path = sample['audio_filepath']
         name = str(path).split('/')[-1].split('.')[0]
         label = sample['accent']
         text = sample['text']
         duration = librosa.get_duration(filename=path)
         if duration > 30:
             continue
         data.append({
             "name": name,
             "path": path,
             "accent": label,
             "text": text
         })
     except Exception as e:
         print(str(path), e)
         pass  

    df = pd.DataFrame(data)
    df['mp3path'] = df['path'].str.replace('\.wav', '.mp3')
    df['mp3path'] = df['mp3path'].str.replace('wav', 'clips')
    print(df.head())
    print(df.shape)
    df.to_csv(json_path+output_file, sep='\t', encoding='utf-8', index=False)

#%%

#%%
def create_file_classifier_cluster(accent, input_file, _type):
    data = []
    home_path = '/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/mz-isca/expts/'
    json_path = os.path.join(home_path, accent, f'manifests/TSS_output/all/budget_200/{_type}/run_1/train/')
    json_file = open(json_path + input_file)
    json_item_list = [line for line in json_file]
    output_file = input_file.replace('.json', f'.csv')
    json_item_list = [json.loads(line.strip()) for line in json_item_list]
    for sample in tqdm(json_item_list):
     try:
         path = sample['audio_filepath']
         text = sample['text']
         duration = librosa.get_duration(filename=path)
         label = sample['dir']
         if duration > 30:
             continue
         data.append({
             "name": "dummy",
             "accent": label,
             "path": path,
             "text": text
         })
     except Exception as e:
         print(str(path), e)
         pass  

    df = pd.DataFrame(data)
    df['mp3path'] = df['path'].str.replace('\.wav', '.mp3')
    df['mp3path'] = df['mp3path'].str.replace('wav', 'clips')
    print(df.head())
    print(df.shape)
    df.to_csv(json_path + output_file, sep='\t', encoding='utf-8', index=False)

#%%

#%%
def create_file_random(accent, input_file):
    data = []
    home_path = '/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/mz-isca/expts/'
    json_path = os.path.join(home_path, accent, f'manifests/TSS_output/all/budget_200/target_20/random/run_1/train/')
    json_file = open(json_path + input_file)
    json_item_list = [line for line in json_file]
    output_file = input_file.replace('.json', f'.csv')
    json_item_list = [json.loads(line.strip()) for line in json_item_list]
    for sample in tqdm(json_item_list):
     try:
         path = sample['audio_filepath']
         text = sample['text']
         duration = librosa.get_duration(filename=path)
         label = sample['accent']
         if duration > 30:
             continue
         data.append({
             "name": "dummy",
             "accent": label,
             "path": path,
             "text": text
         })
     except Exception as e:
         print(str(path), e)
         pass  

    df = pd.DataFrame(data)
    df['mp3path'] = df['path'].str.replace('\.wav', '.mp3')
    df['mp3path'] = df['mp3path'].str.replace('wav', 'clips')
    print(df.head())
    print(df.shape)
    df.to_csv(json_path + output_file, sep='\t', encoding='utf-8', index=False)

#%%

#%%
def create_file_seeddevtest(accent, input_file):
    data = []
    home_path = '/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/mz-isca/expts/'
    json_path = os.path.join(home_path, accent, f'manifests/')
    json_file = open(json_path + input_file)
    json_item_list = [line for line in json_file]
    output_file = input_file.replace('.json', f'.csv')
    json_item_list = [json.loads(line.strip()) for line in json_item_list]
    for sample in tqdm(json_item_list):
     try:
         path = sample['audio_filepath']
         text = sample['text']
         duration = librosa.get_duration(filename=path)
         label = sample['accent']
         if duration > 30:
             continue
         data.append({
             "name": "dummy",
             "accent": label,
             "path": path,
             "text": text
         })
     except Exception as e:
         print(str(path), e)
         pass  

    df = pd.DataFrame(data)
    df['mp3path'] = df['path'].str.replace('\.wav', '.mp3')
    df['mp3path'] = df['mp3path'].str.replace('wav', 'clips')
    print(df.head())
    print(df.shape)
    df.to_csv(json_path + output_file, sep='\t', encoding='utf-8', index=False)

#%%


#%%
accent = 'malaysia'
for fxn in ['FL2MI', 'GCMI', 'LogDMI']:
    create_file_MI(accent, 'train.json', fxn)
#for _type in ['classifer', 'cluster']:
#    create_file_classifier_cluster(accent, 'train.json', _type)
create_file_random(accent, 'train.json')
for fname in ['seed.json', 'dev.json', 'test.json']:
    create_file_seeddevtest(accent, fname) 

#%%


#%%
accents = ['african', 'indian', 'hongkong', 'philippines', 
           'england', 'scotland', 'ireland', 'australia', 
           'canada', 'us']
for accent in accents:
    create_file_classifier_cluster(accent, 'train.json', 'cluster')

#%%
