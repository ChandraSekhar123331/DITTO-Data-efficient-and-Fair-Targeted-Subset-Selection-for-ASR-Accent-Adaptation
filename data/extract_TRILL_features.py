# creates feature TRILL files, currently set for CMU accents
import numpy as np
import pickle
import json
from tqdm import tqdm
import librosa
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
assert tf.executing_eagerly()
import tensorflow_hub as hub
module = hub.load('https://tfhub.dev/google/nonsemantic-speech-benchmark/trill-distilled/2')


base_dir = '/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/CMU_expts/accent/'
dirs = ['arabic', 'korean', 'chinese', 'spanish', 'hindi', 'vietnamese']

def extract_features(file_list, file_dir):
    with open(file_dir.replace('.json', '_TRILL.file'), 'wb') as f:
        for file in tqdm(file_list):
            waveform, sample_rate = librosa.load(file['audio_filepath'], sr=16000)
            features = module(samples=waveform, sample_rate=16000)['embedding']
            features.shape.assert_is_compatible_with([None, 2048])
            pickle.dump(features.numpy().mean(0).reshape(1, -1), f)

for _dir in tqdm(dirs):
    manifests_path = base_dir + _dir + '/'
    print('_'*20)
    print(_dir)

    seed_file_dir = manifests_path + 'seed.json'
    seed_file = open(seed_file_dir)
    seed_list = [json.loads(line.strip()) for line in seed_file]

    selection_file_dir = manifests_path + 'selection.json'
    selection_file = open(selection_file_dir)
    selection_list = [json.loads(line.strip()) for line in selection_file]

    test_file_dir = manifests_path + 'test.json'
    test_file = open(test_file_dir)
    test_list = [json.loads(line.strip()) for line in test_file]

    print('seed_file_starting')
    print(seed_file_dir)
    extract_features(seed_list, seed_file_dir)
    print(len(seed_list))
    print('seed_file_ending ...\n')
    
    print('selection_file_starting')
    extract_features(selection_list, selection_file_dir)
    print(len(selection_list))
    print('selection_file_ending ...\n\n')
    
    print('test_file_starting')
    extract_features(test_list, test_file_dir)
    print(len(test_list))
    print('test_file_ending ...\n\n')