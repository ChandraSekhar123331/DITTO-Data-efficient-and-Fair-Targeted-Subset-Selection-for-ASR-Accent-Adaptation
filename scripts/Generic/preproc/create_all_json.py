import argparse
import json
import os
from pprint import pprint

from utils.dataset import all_datasets, update_config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=all_datasets(), type=str, required=True)
    parser.add_argument("--cuda", type=int, required=True, choices=[0, 1, 2, 3])
    parser.add_argument("--accent", type=str, required=True)
    args = parser.parse_args()
    config = vars(args)
    return update_config(config)

def merge_all(config):
    SETTING_PATH = os.path.join(config["accent"], "")
    OUT_PATH = os.path.join(
        config["FULL_DATASET_PATH"], SETTING_PATH, "all.json"
    )

    SEED_PATH = os.path.join(config["FULL_DATASET_PATH"], SETTING_PATH, "seed.json")
    DEV_PATH = os.path.join(config["FULL_DATASET_PATH"], SETTING_PATH, "dev.json")
    SELECTION_PATH = os.path.join(config["FULL_DATASET_PATH"], SETTING_PATH, "selection.json")
    TEST_PATH = os.path.join(config["FULL_DATASET_PATH"], SETTING_PATH, "test.json")

    with open(OUT_PATH, "w") as OUT_file:
        OUT_LINES = []
        for path in [SEED_PATH, DEV_PATH, SELECTION_PATH, TEST_PATH]:
            OUT_LINES.extend([json.loads(line.strip()) for line in open(path)])
        for line in OUT_LINES:
            OUT_file.write(json.dumps(line))
            OUT_file.write("\n")



if __name__ == "__main__":
    config = get_args()
    pprint(config)

    merge_all(config)


