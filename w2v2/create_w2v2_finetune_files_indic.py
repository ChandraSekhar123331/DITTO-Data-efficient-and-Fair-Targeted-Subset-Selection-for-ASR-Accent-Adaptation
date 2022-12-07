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
def create_file(accent, input_file):
    data = []
    home_path = '../data'
    json_path = os.path.join(home_path, accent, 'manifests/')
    json_file = open(json_path + input_file)
    output_file = input_file.replace('.json', f'.csv')
    json_item_list = [line for line in json_file]
    json_item_list = [json.loads(line.strip()) for line in json_item_list]
    for sample in tqdm(json_item_list):
     try:
         path = sample['audio_filepath']
         text = sample['text']
         duration = sample['duration']
         if duration > 30:
             continue
         data.append({
             "path": path,
             "text": text
         })
     except Exception as e:
         print(str(path), e)
         pass  

    df = pd.DataFrame(data)
    print(df.head())
    print(df.shape)
    df.to_csv(json_path + output_file, sep='\t', encoding='utf-8', index=False)

#%%

#%%
accent = 'hindi_male_english'
#create_file(accent, 'selection.json', 1, 300)
create_file(accent, 'test.json')
create_file(accent, 'dev.json')
#create_file(accent, 'seed.json', 'seed_train.csv')
#create_file(accent, 'dev.json', 'dev.csv')
#create_file(accent, 'test.json', 'test.csv')

#%%
