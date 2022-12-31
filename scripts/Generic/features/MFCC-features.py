import argparse
import json
import os
import pickle
from pprint import pprint

import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
from utils.dataset import all_datasets, all_servers, update_config


def extract_features(file_list, feature_file, config):

    mfcc_transform = T.MFCC(
        sample_rate=22050,
        n_mfcc=39,
        melkwargs={
            "n_fft": 2048,
            "n_mels": 256,
            "hop_length": 512,
            "mel_scale": "htk",
        },
    )
    feature_dict = {}
    for file in tqdm(file_list):
        waveform, sample_rate = torchaudio.load(
            os.path.join(config["WAV_PATH"], file["audio_filepath"])
        )
        mfcc_features = mfcc_transform(waveform).mean(2).detach().numpy()
        feature_dict[file["audio_filepath"]] = mfcc_features

    with open(feature_file, "wb") as f:
        pickle.dump(feature_dict, f)


def generate_features(config):
    SETTING_PATH = os.path.join(
        config["accent"],
        config["INFER_JSON_PATH"],
    )
    JSON_PATH = os.path.join(
        config["FULL_DATASET_PATH"], SETTING_PATH, config["INFER_JSON_NAME"]
    )
    FEATURE_DIR = os.path.join(
        config["FULL_DATASET_PATH"], SETTING_PATH, "features", "MFCC"
    )
    os.makedirs(FEATURE_DIR, exist_ok=True)
    FEATURE_FILE = os.path.join(FEATURE_DIR, config["INFER_JSON_NAME"][:-5] + ".file")

    json_list = [json.loads(line.strip()) for line in open(JSON_PATH)]

    extract_features(json_list, FEATURE_FILE, config)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=all_datasets(), type=str, required=True)
    parser.add_argument("--server", choices=all_servers(), type=str, required=True)
    parser.add_argument("--cuda", type=int, required=True, choices=[0, 1, 2, 3])
    parser.add_argument("--accent", required=True, type=str)
    parser.add_argument("--json_path", type=str, default="")
    parser.add_argument("--json_name", type=str, required=True)
    args = parser.parse_args()
    return update_config(vars(args))


if __name__ == "__main__":
    config = get_args()
    config["INFER_JSON_PATH"] = config["json_path"]
    config["INFER_JSON_NAME"] = config["json_name"]
    pprint(config)
    generate_features(config)
