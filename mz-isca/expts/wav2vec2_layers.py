import os
import numpy as np
import json
from tqdm import tqdm
import pickle
import librosa
import torch
import torchaudio
from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model

device = torch.device('cuda')

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h").to(device)

def extract_features(file_list, file_dir, layer_num):
    with open(file_dir.replace('.json', f'_w2v2_layer-{layer_num}.file'), 'ab') as f:
        for file in tqdm(file_list):
#            waveform, sample_rate = torchaudio.load(file['audio_filepath'])
            waveform, sample_rate = librosa.load(file['audio_filepath'], sr=16000)
            waveform = torch.tensor(waveform)
            speech_inp = tokenizer(waveform, return_tensors='pt', padding='longest').input_values.to(device)
            w2v2_features = model(speech_inp, output_hidden_states=True).hidden_states[layer_num - 1].mean(1).cpu().detach().numpy()
            pickle.dump(w2v2_features, f)

#for i in [1, 5, 10, 15, 20, 25]:
for i in [5]:
    layer_num = i

    dirs = [f.name for f in os.scandir('../MozillaExpts/') if f.is_dir()]
    dirs.remove('.ipynb_checkpoints')
    dirs.remove('bermuda')

    for _dir in dirs:
        print('_'*20)
        print(_dir, i)

    #    seed_file_dir = _dir + '/manifests/seed.json'
    #    seed_file = open(seed_file_dir)
    #    seed_list = [json.loads(line.strip()) for line in seed_file]

        selection_file_dir = _dir + '/manifests/selection.json'
        selection_file = open(selection_file_dir)
        selection_list = [json.loads(line.strip()) for line in selection_file]

    #     test_file_dir = _dir + '/manifests/test.json'
    #     test_file = open(test_file_dir)
    #     test_list = [json.loads(line.strip()) for line in test_file]

    #     print('seed_file_starting')
    #     extract_features(seed_list, seed_file_dir)
    #     print(len(seed_list))
    #     print('seed_file_ending ...\n\n')

        print('selection_file_starting')
        extract_features(selection_list, selection_file_dir, layer_num)
        print(len(selection_list))
        print('selection_file_ending ...\n\n')

    #     print('test_file_starting')
    #     extract_features(test_list, test_file_dir)
    #     print(len(test_list))
    #     print('test_file_ending ...\n\n')
