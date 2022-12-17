import os
from enum import Enum
from math import floor


class Dataset(Enum):
    INDIC = 1
    L2 = 2
    MCV = 3


def all_datasets():
    return tuple(_.name for _ in Dataset)


def get_path(data_obj):
    HOME = HOME_PATH()
    DATASET = DATASET_PATH(data_obj)
    WAV = WAV_PATH(data_obj)
    return HOME, DATASET, WAV


def DATASET_PATH(data_obj):
    if data_obj is Dataset.INDIC:
        PATH = os.path.join("data")
    elif data_obj is Dataset.L2:
        PATH = os.path.join("CMU_expts", "accent-with")
    elif data_obj is Dataset.MCV:
        PATH = os.path.join("mz-isca", "expts")
    else:
        raise ValueError(f"{data_obj.name} is not a valid Dataset")

    return PATH


def HOME_PATH():
    PATH = os.path.join(
        "/home", "mayank", "MTP", "begin_again", "Error-Driven-ASR-Personalization"
    )
    return PATH


def WAV_PATH(data_obj):
    if data_obj is Dataset.INDIC:
        PATH = os.path.join("")
    elif data_obj is Dataset.L2:
        PATH = os.path.join("")
    elif data_obj is Dataset.MCV:
        PATH = os.path.join("mozilla", "cv-corpus-7.0-2021-07-21/en/wav/")
    else:
        raise ValueError(f"{data_obj.name} is not a valid Dataset")

    return PATH


def BUDGET_TO_DURATION(budget):
    return floor(4.92 * budget)


def update_config(config):
    config["HOME_PATH"], config["DATASET_PATH"], config["WAV_PATH"] = get_path(
        Dataset[config["dataset"]]
    )
    config["FULL_DATASET_PATH"] = os.path.join(
        config["HOME_PATH"], config["DATASET_PATH"]
    )
    config["FULL_WAV_PATH"] = os.path.join(config["HOME_PATH"], config["WAV_PATH"])
    return config


if __name__ == "__main__":

    print(Dataset(2).name)
    print(Dataset(2), type(Dataset(2)), Dataset(2).value, Dataset(2).name)
    print(Dataset["INDIC"].value)

    print(Dataset(2) is Dataset["INDIC"])
    print(Dataset(2) is Dataset["L2"])
    print(Dataset(2) is Dataset.L2)

    print(get_path(Dataset["INDIC"]))
    print(get_path(Dataset["L2"]))
    print(get_path(Dataset["MCV"]))

    # print(get_data_path(Dataset['CORAAL'])) # Will raise an error.

    print(all_datasets())
