import os
from enum import Enum
from math import floor

import regex as re


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


def get_accent(line, dataset_name):
    data_obj = Dataset[dataset_name]
    if data_obj is Dataset.INDIC:
        accent = (
            re.search("/indicTTS/([A-Za-z_\-]+)/", line["audio_filepath"])
            .group(1)
            .split("_")[0]
        )
    elif data_obj is Dataset.L2:
        accent_map = {
            "ABA": "Arabic",
            "SKA": "Arabic",
            "YBAA": "Arabic",
            "ZHAA": "Arabic",
            "BWC": "Chinese",
            "LXC": "Chinese",
            "NCC": "Chinese",
            "TXHC": "Chinese",
            "ASI": "Hindi",
            "RRBI": "Hindi",
            "SVBI": "Hindi",
            "TNI": "Hindi",
            "HJK": "Korean",
            "HKK": "Korean",
            "YDCK": "Korean",
            "YKWK": "Korean",
            "EBVS": "Spanish",
            "ERMS": "Spanish",
            "MBMPS": "Spanish",
            "NJS": "Spanish",
            "HQTV": "Vietnamese",
            "PNV": "Vietnamese",
            "THV": "Vietnamese",
            "TLV": "Vietnamese",
        }

        speaker = re.search("/l2_new/([A-Za-z_\-]+)/", line["audio_filepath"]).group(1)
        print(speaker)
        accent = accent_map[speaker]
    elif data_obj is Dataset.MCV:
        accent = line["accent"]
    else:
        raise ValueError(f"{data_obj.name} is not a valid Dataset")
    return accent


def get_all_accents(data_obj):
    if data_obj is Dataset.INDIC:
        accent_list = (
            "assamese",
            "gujarati",
            "hindi",
            "kannada",
            "malayalam",
            "manipuri",
            "rajasthani",
            "tamil",
        )
    elif data_obj is Dataset.L2:
        accent_list = ("arabic", "chinese", "hindi", "korean", "spanish", "vietnamese")
    elif data_obj is Dataset.MCV:
        accent_list = (
            "african",
            "australia",
            "canada",
            "england",
            "hongkong",
            "indian",
            "ireland",
            "philippines",
            "scotland",
            "southatlandtic",
            "us",
        )
    else:
        raise ValueError(f"{data_obj.name} is not a valid Dataset")

    return accent_list


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
    config["all_accents"] = get_all_accents(Dataset[config["dataset"]])
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
