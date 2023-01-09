import argparse
import os
from pprint import pprint

import numpy as np
from utils.dataset import all_datasets, all_servers, update_config
from utils.utils import dump_python_obj, read_python_obj


def extract_unit_norm_features(src_file, dst_file):

    old_features = read_python_obj(src_file)
    assert isinstance(old_features, dict)

    new_features = {}
    for key in old_features:
        new_features[key] = old_features[key] / np.linalg.norm(old_features[key])

    dump_python_obj(new_features, dst_file)


def generate_features(config):
    SETTING_PATH = os.path.join(
        config["accent"],
        config["INFER_JSON_PATH"],
    )
    MFCC_DIR = os.path.join(
        config["FULL_DATASET_PATH"],
        SETTING_PATH,
        "features",
        "MFCC",
    )
    MFCC_UNIT_NORM_DIR = os.path.join(
        config["FULL_DATASET_PATH"], SETTING_PATH, "features", "MFCC_unit_norm"
    )
    os.makedirs(MFCC_UNIT_NORM_DIR, exist_ok=True)
    MFCC_FILE = os.path.join(MFCC_DIR, config["INFER_JSON_NAME"][:-5] + ".file")
    MFCC_UNIT_NORM_FILE = os.path.join(
        MFCC_UNIT_NORM_DIR, config["INFER_JSON_NAME"][:-5] + ".file"
    )

    extract_unit_norm_features(MFCC_FILE, MFCC_UNIT_NORM_FILE)


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
