import argparse
import os
from functools import reduce
from math import gcd as __gcd
from pprint import pprint

import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
from utils.dataset import all_datasets, all_servers, update_config
from utils.test import dump_pretrain_logits as dump_logits
from utils.utils import dump_python_obj, read_lines, read_python_obj


def merge_lines(config, file_name):
    input_lines = [
        read_lines(os.path.join(config["FULL_DATASET_PATH"], accent, file_name))
        for accent in config["query_set"]
    ]
    total_len = reduce(lambda x, y: x + len(y), input_lines, 0)
    print(total_len)
    output_lines = []

    index = 0
    num_accents = len(config["query_set"])
    while len(output_lines) != total_len:
        output_lines.extend(input_lines[index][-config["query_set_composn"][index] :])
        input_lines[index] = input_lines[index][: -config["query_set_composn"][index]]
        index = (index + 1) % num_accents

    return output_lines


def calculate_entropy(logit_array, method):
    # Assumes the logit array is of the form (time * vocab_size)
    vocab_size = 29
    prob = logit_array.T
    assert prob.shape[0] == vocab_size
    entr = entropy(prob)
    assert entr.size == len(logit_array)
    if method == "mean":
        return np.mean(entr)
    elif method == "median":
        return np.median(entr)
    else:
        raise ValueError(f"method = {method} is unknown.")


def dump_pretrain_logits(config):
    for accent in config["all_accents"]:
        INPUT_JSON = os.path.join(config["FULL_DATASET_PATH"], accent, "all.json")
        LOGITS_FILE = os.path.join(
            config["FULL_DATASET_PATH"],
            accent,
            "features",
            "PRETRAIN-LOGITS",
            config["method"],
            "all.logits",
        )
        PRETRAINED_CKPTS = os.path.join(
            config["HOME_PATH"], "models", "pretrained_checkpoints"
        )
        OUTPUT_DIR = os.path.dirname(LOGITS_FILE)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        dump_logits(
            LOGITS_FILE=LOGITS_FILE,
            TEST_JSON_PATH=INPUT_JSON,
            PRETRAINED_CKPTS=PRETRAINED_CKPTS,
            WAV_PATH=config["FULL_WAV_PATH"],
            CUDA=config["cuda"],
            SERVER=config["server"],
        )

        logits = read_python_obj(LOGITS_FILE)
        # print(len(logits))
        # print(logits[0].shape, type(logits))
        lines = read_lines(INPUT_JSON)
        assert len(logits) == len(lines)

        entropy_dict = {}
        for index, line in enumerate(lines):
            entropy_dict[line["audio_filepath"]] = calculate_entropy(
                logits[index], method=config["method"]
            )
        ENTROPY_PATH = os.path.join(os.path.dirname(LOGITS_FILE), "all.file")
        dump_python_obj(entropy_dict, ENTROPY_PATH)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=all_datasets(), type=str, required=True)
    parser.add_argument("--server", choices=all_servers(), type=str, required=True)
    parser.add_argument("--cuda", type=int, required=True, choices=[0, 1, 2, 3])
    parser.add_argument("--method", type=str, required=True, choices=["mean", "median"])
    args = parser.parse_args()
    config = vars(args)
    return update_config(config)


if __name__ == "__main__":
    config = get_args()
    pprint(config)

    dump_pretrain_logits(config)
