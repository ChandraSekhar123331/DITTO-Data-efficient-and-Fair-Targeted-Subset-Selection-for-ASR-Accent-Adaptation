import argparse
import os

import numpy as np
from utils.dataset import BUDGET_TO_DURATION, all_datasets, all_servers, update_config
from utils.finetune import finetune
from utils.test import test
from utils.utils import dump_lines, read_lines, sample_greedy


def get_metric(line, config):
    return line["CER"] if config["CER"] else line["WER"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=all_datasets(), type=str, required=True)
    parser.add_argument("--server", choices=all_servers(), type=str, required=True)
    parser.add_argument("--cuda", type=int, required=True, choices=[0, 1, 2, 3])
    parser.add_argument("--sample", action="store_true", default=False)
    parser.add_argument("--finetune", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--accent", required=True, type=str)
    parser.add_argument("--json_path", default="", type=str)
    parser.add_argument("--json_name", type=str, required=True)
    parser.add_argument("--CER", action="store_true", default=False)
    args = parser.parse_args()
    config = vars(args)
    return update_config(config)


def sample_true_error(config):

    if not config["sample"]:
        return
    print("Sampling True {} model".format({"CER" if config["CER"] else "WER"}))

    for run in range(1, 4):
        INPUT_SETTING_PATH = os.path.join(config["accent"], config["INFER_JSON_PATH"])
        INPUT_JSON_PATH = os.path.join(
            config["FULL_DATASET_PATH"], INPUT_SETTING_PATH, config["INFER_JSON_NAME"]
        )
        OUTPUT_JSON_PATH = os.path.join(
            config["FULL_DATASET_PATH"],
            INPUT_SETTING_PATH,
            "budget_{}".format(config["budget"]),
            f"true_{'CER' if config['CER'] else 'WER'}",
            f"run_{run}",
            "train.json",
        )

        lines = read_lines(INPUT_JSON_PATH)
        rng = np.random.default_rng(seed=run)
        rng.shuffle(lines)

        lines.sort(key=lambda x: get_metric(x, config), reverse=True)

        selected_lines = sample_greedy(lines, BUDGET_TO_DURATION(config["budget"]))

        dump_lines(selected_lines, OUTPUT_JSON_PATH)


def finetune_true_error(config):
    if not config["finetune"]:
        return
    print("Finetuning True {} model".format({"CER" if config["CER"] else "WER"}))

    PRETRAINED_CKPTS = os.path.join(
        config["HOME_PATH"], "models", "pretrained_checkpoints"
    )
    for run in range(1, 4):
        SETTING_PATH = os.path.join(
            config["accent"],
            config["INFER_JSON_PATH"],
            "budget_{}".format(config["budget"]),
            f"true_{'CER' if config['CER'] else 'WER'}",
            f"run_{run}",
        )
        JSON_PATH = os.path.join(
            config["FULL_DATASET_PATH"], SETTING_PATH, "train.json"
        )
        CKPT_PATH = os.path.join(
            PRETRAINED_CKPTS, "quartznet", "finetuned", config["dataset"], SETTING_PATH
        )
        VAL_JSON_PATH = os.path.join(
            config["FULL_DATASET_PATH"], config["accent"], "dev.json"
        )

        finetune(
            JSON_PATH=JSON_PATH,
            VAL_JSON_PATH=VAL_JSON_PATH,
            CKPT_PATH=CKPT_PATH,
            PRETRAINED_CKPTS=PRETRAINED_CKPTS,
            WAV_PATH=config["FULL_WAV_PATH"],
            CUDA=config["cuda"],
            SERVER=config["server"],
        )


def test_true_error(config):
    if not config["test"]:
        return
    print("Testing True {} model".format({"CER" if config["CER"] else "WER"}))

    PRETRAINED_CKPTS = os.path.join(
        config["HOME_PATH"], "models", "pretrained_checkpoints"
    )
    for run in range(1, 4):
        SETTING_PATH = os.path.join(
            config["accent"],
            config["INFER_JSON_PATH"],
            "budget_{}".format(config["budget"]),
            f"true_{'CER' if config['CER'] else 'WER'}",
            f"run_{run}",
        )
        JSON_PATH = os.path.join(
            config["FULL_DATASET_PATH"], SETTING_PATH, "train.json"
        )
        CKPT_PATH = os.path.join(
            PRETRAINED_CKPTS, "quartznet", "finetuned", config["dataset"], SETTING_PATH
        )
        TEST_JSON_PATH = os.path.join(
            config["FULL_DATASET_PATH"], config["accent"], "test.json"
        )

        test(
            JSON_PATH=JSON_PATH,
            TEST_JSON_PATH=TEST_JSON_PATH,
            CKPT_PATH=CKPT_PATH,
            PRETRAINED_CKPTS=PRETRAINED_CKPTS,
            WAV_PATH=config["FULL_WAV_PATH"],
            CUDA=config["cuda"],
            SERVER=config["server"],
        )


if __name__ == "__main__":
    config = get_args()
    config["INFER_JSON_PATH"] = config["json_path"]
    config["INFER_JSON_NAME"] = config["json_name"]

    sample_true_error(config)
    finetune_true_error(config)
    test_true_error(config)
