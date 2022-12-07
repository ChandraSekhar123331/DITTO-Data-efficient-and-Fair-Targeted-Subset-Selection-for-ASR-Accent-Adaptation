import os
import numpy as np
import pickle
import json
import torch
from tqdm import tqdm
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

mfcc_transform = T.MFCC(
    sample_rate=48000,
    n_mfcc=39,
    melkwargs={
      'n_fft': 2048,
      'n_mels': 256,
      'hop_length': 512,
      'mel_scale': 'htk',
    }
)

def extract_features(file_list, file_dir):
    with open(file_dir.replace('.json', '_39.file'), 'ab') as f:
        for file in tqdm(file_list):
            waveform, sample_rate = torchaudio.load(file['audio_filepath'])
            mfcc_features = mfcc_transform(waveform).mean(2).detach().numpy()
            pickle.dump(mfcc_features, f)

jsons_path = 'jsons/'
jsons = [f.name for f in os.scandir(jsons_path) if 'json' in f.name and f.name.split('.')[0] not in ['unlabelled', 'other']]

print(jsons)
for accent in jsons:
    print('_'*20)
    print(accent)
    
    json_file_path = jsons_path + accent 
    json_file = open(json_file_path)
    json_list = [json.loads(line.strip()) for line in json_file]
    
    print('file_starting')
    extract_features(json_list, json_file_path)
    print(len(json_list))
    print('file_ending ...\n\n')
