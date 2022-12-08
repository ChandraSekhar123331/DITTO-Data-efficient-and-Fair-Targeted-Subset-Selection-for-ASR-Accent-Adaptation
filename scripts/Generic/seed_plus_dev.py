import argparse
from pprint import pprint
from dataset import get_path, all_datasets, Dataset
import os
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=all_datasets(), type=str, required=True)
    parser.add_argument("--cuda", type=int, required=True, choices=[0, 1, 2, 3])
    parser.add_argument("--accent", type=str, required=True)
    args = parser.parse_args()    
    return vars(args)


def merge_seed_dev(config):
    SETTING_PATH = os.path.join(config["accent"], "")
    OUT_PATH = os.path.join(config["FULL_DATASET_PATH"], SETTING_PATH, "seed_plus_dev.json")

    SEED_PATH = os.path.join(config["FULL_DATASET_PATH"], SETTING_PATH, "seed.json")
    DEV_PATH = os.path.join(config["FULL_DATASET_PATH"], SETTING_PATH, "dev.json")

    with open(OUT_PATH, 'w') as OUT_file:
        SEED_lines = [json.loads(line.strip()) for line in open(SEED_PATH)]
        DEV_lines = [json.loads(line.strip()) for line in open(DEV_PATH)]
        
        OUT_lines = SEED_lines + DEV_lines
        for line in OUT_lines:
            OUT_file.write(json.dumps(line))
            OUT_file.write("\n")
    


if __name__ == "__main__":
    config = get_args()
    config["HOME_PATH"], config["DATASET_PATH"], config["WAV_PATH"] = get_path(Dataset[config["dataset"]])
    config["FULL_DATASET_PATH"] = os.path.join(config["HOME_PATH"], config["DATASET_PATH"])
    config["FULL_WAV_PATH"] = os.path.join(config["HOME_PATH"], config["WAV_PATH"])
    config["FULL_PATH"] = os.path.join(config["HOME_PATH"], config["DATASET_PATH"], config["accent"])
    pprint(config)

    merge_seed_dev(config)