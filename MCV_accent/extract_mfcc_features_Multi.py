import os
import numpy as np
import pickle
import json
import torch
from tqdm import tqdm
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from multiprocessing import Pool, Manager

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

def extract_features(file_dir):
    json_file = open(file_dir)
    file_list = [json.loads(line.strip()) for line in json_file]
    file_list = file_list[:1000]

    with open(file_dir.replace('.json', '_39.file'), 'ab') as f:
        for file in tqdm(file_list):
            waveform, sample_rate = torchaudio.load(file['audio_filepath'])
            mfcc_features = mfcc_transform(waveform).mean(2).detach().numpy()
            pickle.dump(mfcc_features, f)

jsons_path = 'jsons/'
jsons = [jsons_path + f.name for f in os.scandir(jsons_path) if 'json' in f.name and f.name.split('.')[0] not in ['unlabelled', 'other', 'philippines.json', 'wales.json', 'scotland.json', 'hongkong.json', 'malaysia.json', 'indian.json', 'australia.json']]
print(jsons)

manager = Manager()
L = manager.list()
pool = Pool(8)
pool.map(extract_features, jsons)
pool.close()
pool.join()

