import argparse
import os
import shutil
from pprint import pprint

from utils.dataset import BUDGET_TO_DURATION, all_datasets, all_servers, update_config
from utils.finetune import finetune
from utils.test import test
from utils.utils import dump_lines, read_lines, read_python_obj, sample_greedy


def sample_global_entropy(config):
    if not config["sample"]:
        return

    print("sampling global entropy")

    ground_list = []
    entropy_dict = {}

    for acc in config["all_accents"]:
        JSON_PATH = os.path.join(config["FULL_DATASET_PATH"], acc, "selection.json")
        ENTROPY_FEATURES_PATH = os.path.join(
            config["FULL_DATASET_PATH"],
            acc,
            "features",
            "PRETRAIN-LOGITS",
            config["agg"],
            "all.file",
        )
        entropy_vals = read_python_obj(ENTROPY_FEATURES_PATH)
        entropy_dict.update(entropy_vals)
        ground_list.extend(read_lines(JSON_PATH))

    ground_list.sort(
        key=lambda line: entropy_dict[line["audio_filepath"]], reverse=True
    )
    print(
        "Top 10 entropies are: ",
        [entropy_dict[line["audio_filepath"]] for line in ground_list[:10]],
    )

    for run in range(1, 4):

        selected_lines = sample_greedy(
            ground_list, BUDGET_TO_DURATION(config["budget"])
        )

        OUTPUT_JSON_PATH = os.path.join(
            config["FULL_DATASET_PATH"],
            "results",
            f"budget_{config['budget']}",
            "global_entropy",
            "agg_{}".format(config["agg"]),
            f"run_{run}",
            "train.json",
        )
        dump_lines(selected_lines, OUTPUT_JSON_PATH)


def finetune_global_entropy(config):
    if not config["finetune"]:
        return
    print(
        "finetuning global entropy with accent = {}".format(config["finetune_accent"])
    )

    PRETRAINED_CKPTS = os.path.join(
        config["HOME_PATH"], "models", "pretrained_checkpoints"
    )

    for run in range(1, 4):
        OLD_JSON_PATH = os.path.join(
            config["FULL_DATASET_PATH"],
            "results",
            f"budget_{config['budget']}",
            "global_entropy",
            "agg_{}".format(config["agg"]),
            f"run_{run}",
            "train.json",
        )

        SETTING_PATH = os.path.join(
            config["finetune_accent"],
            "results",
            f"budget_{config['budget']}",
            "global_entropy",
            "agg_{}".format(config["agg"]),
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


def test_global_entropy(config):
    if not config["test"]:
        return
    print("testing global entropy with accent = {}".format(config["test_accent"]))
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
            "global_entropy",
            "agg_{}".format(config["agg"]),
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

    parser.add_argument("--finetune", action="store_true", default=False)
    parser.add_argument("--finetune_accent", type=str, default="")

    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--test_accent", type=str, default="")

    parser.add_argument("--sample", action="store_true", default=False)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--agg", type=str, choices=["mean", "median"], required=True)

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
    pprint(config)
    sample_global_entropy(config)
    finetune_global_entropy(config)
    test_global_entropy(config)
