import argparse
import os
from functools import reduce
from math import gcd as __gcd
from pprint import pprint

from utils.dataset import all_datasets, update_config
from utils.utils import dump_lines, read_lines


def merge_lines(config, file_name):
    input_lines = [read_lines(os.path.join(config["FULL_DATASET_PATH"], accent, file_name)) for accent in config["query_set"]]
    total_len = reduce(lambda x, y: x + len(y), input_lines, 0)
    print(total_len)
    output_lines = []

    index = 0
    num_accents = len(config["query_set"])
    while len(output_lines) != total_len:
        output_lines.extend(input_lines[index][-config["query_set_composn"][index]:])
        input_lines[index] = input_lines[index][:-config["query_set_composn"][index]]
        index = (index + 1) % num_accents

    return output_lines





def create_mixed_set(config):
    name = "-".join(config["query_set"])
    compsn = "-".join(map(str, config["query_set_composn"]))
    full_name = f"{name}::{compsn}"
    OUTPUT_DIR = os.path.join(config["FULL_DATASET_PATH"], "mixed", full_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    for filename in ["seed.json", "dev.json", "selection.json", "test.json", "seed_plus_dev.json"]:
        lines = merge_lines(config, filename)
        dump_lines(lines, os.path.join(OUTPUT_DIR, filename))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=all_datasets(), type=str, required=True)
    parser.add_argument("--cuda", type=int, required=True, choices=[0, 1, 2, 3])
    parser.add_argument("--query_set", nargs="*", type=str, required=True)
    parser.add_argument("--query_set_composn", nargs="*", type=int, required=True)
    args = parser.parse_args()
    config = vars(args)
    if len(config["query_set"]) != len(config["query_set_composn"]):
        pprint(config)
        raise ValueError("length of query set and query set composn must be same")

    gcd = reduce(lambda x, y: __gcd(x, y), config["query_set_composn"], 0)
    config["query_set_composn"] = list(map(lambda x: x//gcd, config["query_set_composn"]))

    return update_config(config)


if __name__ == "__main__":
    config = get_args()
    pprint(config)

    create_mixed_set(config)