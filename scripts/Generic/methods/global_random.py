import argparse
import os
import shutil

from utils.dataset import BUDGET_TO_DURATION, all_datasets, all_servers, update_config
from utils.finetune import finetune
from utils.test import test
from utils.utils import dump_lines, read_lines, sample_random


def sample_global_random(config):
    if not config["sample"]:
        return

    print("sampling global random")

    ground_list = []

    for acc in config["all_accents"]:
        JSON_PATH = os.path.join(config["FULL_DATASET_PATH"], acc, "selection.json")
        ground_list.extend(read_lines(JSON_PATH))

    for run in range(1, 4):

        selected_lines = sample_random(
            ground_list, BUDGET_TO_DURATION(config["budget"]), seed=run
        )

        OUTPUT_JSON_PATH = os.path.join(
            config["FULL_DATASET_PATH"],
            "results",
            f"budget_{config['budget']}",
            "global_random",
            f"run_{run}",
            "train.json",
        )
        dump_lines(selected_lines, OUTPUT_JSON_PATH)


def finetune_global_random(config):
    if not config["finetune"]:
        return
    print("finetuning global random with accent = {}".format(config["finetune_accent"]))

    PRETRAINED_CKPTS = os.path.join(
        config["HOME_PATH"], "models", "pretrained_checkpoints"
    )

    for run in range(1, 4):
        OLD_JSON_PATH = os.path.join(
            config["FULL_DATASET_PATH"],
            "results",
            f"budget_{config['budget']}",
            "global_random",
            f"run_{run}",
            "train.json",
        )
        SETTING_PATH = os.path.join(
            config["finetune_accent"],
            "results",
            f"budget_{config['budget']}",
            "global_random",
            f"run_{run}",
        )

        NEW_JSON_PATH = os.path.join(
            config["FULL_DATASET_PATH"], SETTING_PATH, "train.json"
        )
        os.makedirs(os.path.dirname(NEW_JSON_PATH), exist_ok=True)
        shutil.copy(src=OLD_JSON_PATH, dst=NEW_JSON_PATH)

        VAL_JSON_PATH = os.path.join(
            config["FULL_DATASET_PATH"], config["finetune_accent"], "dev.json"
        )
        CKPT_PATH = os.path.join(
            PRETRAINED_CKPTS, "quartznet", "finetuned", config["dataset"], SETTING_PATH
        )
        finetune(
            JSON_PATH=NEW_JSON_PATH,
            VAL_JSON_PATH=VAL_JSON_PATH,
            CKPT_PATH=CKPT_PATH,
            PRETRAINED_CKPTS=PRETRAINED_CKPTS,
            WAV_PATH=config["FULL_WAV_PATH"],
            CUDA=config["cuda"],
            SERVER=config["server"],
        )


def test_global_random(config):
    if not config["test"]:
        return
    print("testing global random with accent = {}".format(config["test_accent"]))
    PRETRAINED_CKPTS = os.path.join(
        config["HOME_PATH"], "models", "pretrained_checkpoints"
    )

    for run in range(1, 4):
        TEST_JSON_PATH = os.path.join(
            config["FULL_DATASET_PATH"], config["test_accent"], "test.json"
        )
        SETTING_PATH = os.path.join(
            config["test_accent"],
            "results",
            f"budget_{config['budget']}",
            "global_random",
            f"run_{run}",
        )
        DUMP_PATH = os.path.join(config["FULL_DATASET_PATH"], SETTING_PATH)
        CKPT_PATH = os.path.join(
            PRETRAINED_CKPTS, "quartznet", "finetuned", config["dataset"], SETTING_PATH
        )
        test(
            DUMP_PATH=DUMP_PATH,
            TEST_JSON_PATH=TEST_JSON_PATH,
            CKPT_PATH=CKPT_PATH,
            PRETRAINED_CKPTS=PRETRAINED_CKPTS,
            WAV_PATH=config["FULL_WAV_PATH"],
            CUDA=config["cuda"],
            SERVER=config["server"],
        )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=all_datasets(), type=str, required=True)
    parser.add_argument("--server", choices=all_servers(), type=str, required=True)
    parser.add_argument("--cuda", type=int, required=True, choices=[0, 1, 2, 3])
    parser.add_argument("--sample", action="store_true", default=False)
    parser.add_argument("--finetune", action="store_true", default=False)
    parser.add_argument("--finetune_accent", type=str, default="")
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--test_accent", type=str, default="")
    parser.add_argument("--budget", type=int, required=True)
    args = parser.parse_args()
    config = vars(args)

    if config["test"] and not config["test_accent"]:
        raise ValueError("test_accent can't be empty when --test switch is used")

    if config["finetune"] and not config["finetune_accent"]:
        raise ValueError(
            "finetune_accent can't be empty when --finetune switch is used"
        )

    return update_config(config)


if __name__ == "__main__":
    config = get_args()

    sample_global_random(config)
    finetune_global_random(config)
    test_global_random(config)
