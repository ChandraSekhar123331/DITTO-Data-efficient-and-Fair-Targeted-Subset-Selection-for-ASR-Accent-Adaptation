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
def create_file(_dir, accent, input_file):
    data = []
    json_path = os.path.join(_dir, accent)
    json_file = open(json_path + f'/{input_file}')
    json_item_list = [line for line in json_file]
    output_file = input_file.replace('.json', f'.csv')
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
    df.to_csv(json_path + f'/{output_file}', sep='\t', encoding='utf-8', index=False)

#%%

#%%
_dir = 'tamil_kannada'
accent = 'tamil_kannada_N_2'
create_file(_dir, accent, 'train.json')
create_file(_dir, accent, 'dev.json')

#%%
