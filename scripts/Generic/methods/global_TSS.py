import argparse
import os
import shutil
from collections import Counter
from pprint import pprint

from utils.dataset import (
    BUDGET_TO_DURATION,
    all_datasets,
    all_features,
    all_servers,
    update_config,
)
from utils.finetune import finetune
from utils.test import test
from utils.utils import (
    build_kernel,
    build_SMI_OBJ,
    dump_lines,
    dump_python_obj,
    get_accent,
    load_features,
    maximise_SMI,
    read_lines,
    sample_greedy,
)


def sample_global_TSS(config):
    if not config["sample"]:
        return

    print("sampling global TSS, target = {}".format(config["target_accent"]))

    ground_list = []
    for acc in config["all_accents"]:
        JSON_PATH = os.path.join(config["FULL_DATASET_PATH"], acc, "selection.json")
        ground_list.extend(read_lines(JSON_PATH))
    ground_features = load_features(
        ground_list, config["dataset"], config["FULL_DATASET_PATH"], config["feature"]
    )

    QUERY_JSON_PATH = os.path.join(
        config["FULL_DATASET_PATH"],
        config["target_directory_path"],
        config["target_accent"],
        "seed.json",
    )
    query_list = read_lines(QUERY_JSON_PATH)[: config["target"]]
    query_features = load_features(
        query_list, config["dataset"], config["FULL_DATASET_PATH"], config["feature"]
    )

    print(
        "ground_list: ",
        Counter([get_accent(line, config["dataset"]) for line in ground_list]),
    )
    print(
        "query_list: ",
        Counter([get_accent(line, config["dataset"]) for line in query_list]),
    )

    print("Building kernels")
    ground_ground_kernel = build_kernel(ground_features, ground_features, config["sim"])
    print("ground_ground_kernel.shape: ", ground_ground_kernel.shape)
    query_ground_kernel = build_kernel(query_features, ground_features, config["sim"])
    print("query_ground_kernel.shape: ", query_ground_kernel.shape)
    query_query_kernel = build_kernel(query_features, query_features, config["sim"])
    print("query_query_kernel.shape: ", query_query_kernel.shape)
    print("Building SMI objects")
    SMI_obj = build_SMI_OBJ(
        ground_ground=ground_ground_kernel,
        query_ground=query_ground_kernel,
        query_query=query_query_kernel,
        fxn=config["fxn"],
        eta=config["eta"],
    )
    print("Maximising SMI objective")
    SMI_output = maximise_SMI(SMI_obj, budget=2 * config["budget"])
    SMI_indices = [_[0] for _ in SMI_output]
    SMI_lines = [ground_list[index] for index in SMI_indices]
    # print(Counter([get_accent(line, config["dataset"]) for line in SMI_lines]))
    SMI_gains = [_[1] for _ in SMI_output]
    # print(SMI_indices)
    # print(SMI_gains)

    selected_lines = sample_greedy(SMI_lines, BUDGET_TO_DURATION(config["budget"]))
    print(
        "selected_lines: ",
        Counter([get_accent(line, config["dataset"]) for line in selected_lines]),
    )
    selected_gains = SMI_gains[: len(selected_lines)]

    for run in range(1, 4):
        SETTING_PATH = os.path.join(
            config["target_directory_path"],
            config["target_accent"],
            "results",
            f"budget_{config['budget']}",
            "global-TSS",
            "target_{}".format(config["target"]),
            "fxn_{}".format(config["fxn"]),
            "feature_{}".format(config["feature"]),
            "sim_{}".format(config["sim"]),
            "eta_{}".format(config["eta"]),
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


def finetune_global_TSS(config):
    if not config["finetune"]:
        return
    print(
        "finetuning TSS-target = {} with accent = {}".format(
            config["target_accent"], config["finetune_accent"]
        )
    )

    PRETRAINED_CKPTS = os.path.join(
        config["HOME_PATH"], "models", "pretrained_checkpoints"
    )

    for run in range(1, 4):
        OLD_SETTING_PATH = os.path.join(
            config["target_directory_path"],
            config["target_accent"],
            "results",
            f"budget_{config['budget']}",
            "global-TSS",
            "target_{}".format(config["target"]),
            "fxn_{}".format(config["fxn"]),
            "feature_{}".format(config["feature"]),
            "sim_{}".format(config["sim"]),
            "eta_{}".format(config["eta"]),
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
            config["target_directory_path"],
            config["target_accent"],
            "global-TSS",
            "target_{}".format(config["target"]),
            "fxn_{}".format(config["fxn"]),
            "feature_{}".format(config["feature"]),
            "sim_{}".format(config["sim"]),
            "eta_{}".format(config["eta"]),
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


def test_global_TSS(config):
    if not config["test"]:
        return
    print(
        "testing global TSS-target = {} with accent = {}".format(
            config["target_accent"], config["test_accent"]
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
            config["target_directory_path"],
            config["target_accent"],
            "global-TSS",
            "target_{}".format(config["target"]),
            "fxn_{}".format(config["fxn"]),
            "feature_{}".format(config["feature"]),
            "sim_{}".format(config["sim"]),
            "eta_{}".format(config["eta"]),
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
    parser.add_argument(
        "--fxn", type=str, required=True, choices=["FL2MI", "GCMI", "LogDMI", "FL1MI"]
    )
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument(
        "--sim", type=str, required=True, choices=["euclidean", "cosine"]
    )
    parser.add_argument(
        "--feature",
        required=True,
        choices=all_features(),
    )
    parser.add_argument("--target", required=True, type=int, choices=[20, 50])
    parser.add_argument("--target_accent", required=True, type=str)
    parser.add_argument("--target_directory_path", type=str, default="")

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
    sample_global_TSS(config)
    finetune_global_TSS(config)
    test_global_TSS(config)
