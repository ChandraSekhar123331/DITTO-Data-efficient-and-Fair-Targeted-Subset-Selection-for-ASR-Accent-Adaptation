import argparse
import os
import shutil
from collections import Counter
from pprint import pprint

from utils.dataset import BUDGET_TO_DURATION, all_datasets, all_servers, update_config
from utils.finetune import finetune
from utils.test import test
from utils.utils import (
    build_kernel,
    build_SM_OBJ,
    dump_lines,
    dump_python_obj,
    get_accent,
    load_features,
    maximise_SM,
    read_lines,
    sample_greedy,
)


def sample_global_SM(config):
    if not config["sample"]:
        return

    print("sampling global SubModular(i.e. SM)")

    ground_list = []
    for acc in config["all_accents"]:
        JSON_PATH = os.path.join(config["FULL_DATASET_PATH"], acc, "selection.json")
        ground_list.extend(read_lines(JSON_PATH))
    print("Loading Features", flush=True)
    ground_features = load_features(
        ground_list, config["dataset"], config["FULL_DATASET_PATH"], config["feature"]
    )

    print(
        "ground_list: ",
        Counter([get_accent(line, config["dataset"]) for line in ground_list]),
    )
    print("Building Kernel", flush=True)
    ground_ground_kernel = build_kernel(
        ground_features,
        ground_features,
        similarity=config["sim"],
    )

    print("Building SM_obj")
    SM_obj = build_SM_OBJ(
        features=ground_features,
        fxn=config["fxn"],
        lambdaVal=config["lambdaVal"],
        kernel=ground_ground_kernel,
        sim=None,
    )
    print("Optimising SM_obj", flush=True)
    SM_result = maximise_SM(SM_obj, config["budget"] * 2)

    SM_indices = [_[0] for _ in SM_result]
    SM_gains = [_[1] for _ in SM_result]

    SMI_lines = [ground_list[index] for index in SM_indices]
    selected_lines = sample_greedy(SMI_lines, BUDGET_TO_DURATION(config["budget"]))
    print(
        "selected_lines: ",
        Counter([get_accent(line, config["dataset"]) for line in selected_lines]),
    )
    selected_gains = SM_gains[: len(selected_lines)]

    for run in range(1, 4):
        SETTING_PATH = os.path.join(
            "results",
            f"budget_{config['budget']}",
            "global-SM",
            "fxn_{}".format(config["fxn"]),
            "feature_{}".format(config["feature"]),
            "sim_{}".format(config["sim"]),
            "lambdaVal_{}".format(config["lambdaVal"]),
            f"run_{run}",
        )
        OUTPUT_JSON_PATH = os.path.join(
            config["FULL_DATASET_PATH"],
            SETTING_PATH,
            "train.json",
        )

        OUTPUT_GAINS_PATH = os.path.join(os.path.dirname(OUTPUT_JSON_PATH), "gains.pkl")
        dump_lines(selected_lines, OUTPUT_JSON_PATH)
        dump_python_obj(selected_gains, OUTPUT_GAINS_PATH)


def finetune_global_SM(config):
    if not config["finetune"]:
        return
    print(
        "finetuning global SubModular(i.e. SM) with accent = {}".format(
            config["finetune_accent"]
        )
    )

    PRETRAINED_CKPTS = os.path.join(
        config["HOME_PATH"], "models", "pretrained_checkpoints"
    )

    for run in range(1, 4):
        OLD_SETTING_PATH = os.path.join(
            "results",
            f"budget_{config['budget']}",
            "global-SM",
            "fxn_{}".format(config["fxn"]),
            "feature_{}".format(config["feature"]),
            "sim_{}".format(config["sim"]),
            "lambdaVal_{}".format(config["lambdaVal"]),
            f"run_{run}",
        )
        OLD_JSON_PATH = os.path.join(
            config["FULL_DATASET_PATH"],
            OLD_SETTING_PATH,
            "train.json",
        )
        NEW_SETTING_PATH = os.path.join(
            config["finetune_accent"],
            "results",
            f"budget_{config['budget']}",
            "global-SM",
            "fxn_{}".format(config["fxn"]),
            "feature_{}".format(config["feature"]),
            "sim_{}".format(config["sim"]),
            "lambdaVal_{}".format(config["lambdaVal"]),
            f"run_{run}",
        )

        NEW_JSON_PATH = os.path.join(
            config["FULL_DATASET_PATH"], NEW_SETTING_PATH, "train.json"
        )
        os.makedirs(os.path.dirname(NEW_JSON_PATH), exist_ok=True)
        shutil.copy(src=OLD_JSON_PATH, dst=NEW_JSON_PATH)

        VAL_JSON_PATH = os.path.join(
            config["FULL_DATASET_PATH"], config["finetune_accent"], "dev.json"
        )
        CKPT_PATH = os.path.join(
            PRETRAINED_CKPTS,
            "quartznet",
            "finetuned",
            config["dataset"],
            NEW_SETTING_PATH,
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


def test_global_SM(config):
    if not config["test"]:
        return
    print(
        "testing global SubModular(i.e. SM) with accent = {}".format(
            config["test_accent"]
        )
    )
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
            "global-SM",
            "fxn_{}".format(config["fxn"]),
            "feature_{}".format(config["feature"]),
            "sim_{}".format(config["sim"]),
            "lambdaVal_{}".format(config["lambdaVal"]),
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
    parser.add_argument("--fxn", type=str, required=True, choices=["FacLoc", "LogDet"])
    parser.add_argument(
        "--sim", type=str, required=True, choices=["euclidean", "cosine"]
    )
    parser.add_argument("--lambdaVal", type=float, required=True)
    parser.add_argument(
        "--feature",
        required=True,
        choices=["MFCC", "w2v2_Ftill10_768-512-256"],
    )

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
    sample_global_SM(config)
    finetune_global_SM(config)
    test_global_SM(config)
