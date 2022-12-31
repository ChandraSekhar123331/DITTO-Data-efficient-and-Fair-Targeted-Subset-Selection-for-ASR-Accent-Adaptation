import argparse
import os
from pprint import pprint

import numpy as np
from tqdm import tqdm
from utils.dataset import all_datasets, all_servers, update_config
from utils.utils import dump_python_obj, read_lines, read_python_list_obj

# def extract_features(file_list, feature_file, config):

#     mfcc_transform = T.MFCC(
#         sample_rate=22050,
#         n_mfcc=39,
#         melkwargs={
#             "n_fft": 2048,
#             "n_mels": 256,
#             "hop_length": 512,
#             "mel_scale": "htk",
#         },
#     )
#     feature_dict = {}
#     for file in tqdm(file_list):
#         waveform, sample_rate = torchaudio.load(
#             os.path.join(config["WAV_PATH"], file["audio_filepath"])
#         )
#         mfcc_features = mfcc_transform(waveform).mean(2).detach().numpy()
#         feature_dict[file["audio_filepath"]] = mfcc_features

#     with open(feature_file, "wb") as f:
#         pickle.dump(feature_dict, f)


def update_feature_file(config):
    FEATURE_SETTING_PATH = os.path.join(
        config["accent"], "features", config["feature_type"]
    )
    FEATURE_PATH = os.path.join(
        config["FULL_DATASET_PATH"], FEATURE_SETTING_PATH, "all.file"
    )
    JSON_PATH = os.path.join(config["FULL_DATASET_PATH"], config["accent"], "all.json")
    os.makedirs(os.path.dirname(FEATURE_PATH), exist_ok=True)

    lines = read_lines(JSON_PATH)
    features = read_python_list_obj(FEATURE_PATH)
    assert isinstance(features, list)
    assert isinstance(features[0], np.ndarray)
    assert len(lines) == len(features)

    dump_python_obj(features, os.path.join(os.path.dirname(FEATURE_PATH), "temp.file"))
    # print(len(lines), len(features))

    feature_dict = {}

    for ind, line in enumerate(lines):
        feature = features[ind]
        feature_dict[line["audio_filepath"]] = feature

    NEW_FEATURE_PATH = os.path.join(os.path.dirname(FEATURE_PATH), "all.file")
    dump_python_obj(feature_dict, NEW_FEATURE_PATH)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=all_datasets(), type=str, required=True)
    parser.add_argument("--server", choices=all_servers(), type=str, required=True)
    parser.add_argument("--cuda", type=int, required=True, choices=[0, 1, 2, 3])
    parser.add_argument("--accent", required=True, type=str)
    parser.add_argument("--feature_type", type=str, required=True)
    args = parser.parse_args()
    return update_config(vars(args))


if __name__ == "__main__":
    config = get_args()
    pprint(config)
    update_feature_file(config)
