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


def all_features():
    return ("MFCC", "MFCC_unit_norm", "w2v2_Ftill10_768-512-256")


def all_servers():
    return ("SWARA", "DRISTI", "AIRAWAT", "DHVANI")


def get_path(data_obj, server):
    HOME = HOME_PATH(server=server)
    DATASET = DATASET_PATH(data_obj)
    WAV = WAV_PATH(data_obj, server=server)
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
            "ABA": "arabic",
            "SKA": "arabic",
            "YBAA": "arabic",
            "ZHAA": "arabic",
            "BWC": "chinese",
            "LXC": "chinese",
            "NCC": "chinese",
            "TXHC": "chinese",
            "ASI": "hindi",
            "RRBI": "hindi",
            "SVBI": "hindi",
            "TNI": "hindi",
            "HJK": "korean",
            "HKK": "korean",
            "YDCK": "korean",
            "YKWK": "korean",
            "EBVS": "spanish",
            "ERMS": "spanish",
            "MBMPS": "spanish",
            "NJS": "spanish",
            "HQTV": "vietnamese",
            "PNV": "vietnamese",
            "THV": "vietnamese",
            "TLV": "vietnamese",
        }

        speaker = re.search("/l2_new/([A-Za-z_\-]+)/", line["audio_filepath"]).group(1)
        # print(speaker)
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


def HOME_PATH(server):
    if server == "SWARA":
        PATH = os.path.join(
            "/home", "mayank", "MTP", "begin_again", "Error-Driven-ASR-Personalization"
        )
    elif server == "AIRAWAT":
        PATH = os.path.join(
            "/home", "mohith", "mayank", "Error-Driven-ASR-Personalization"
        )
    elif server == "DRISTI":
        PATH = os.path.join(
            "/home", "aman", "mayank", "Error-Driven-ASR-Personalization"
        )
    elif server == "DHVANI":
        PATH = os.path.join(
            "/home", "mayur", "mayank", "Error-Driven-ASR-Personalization"
        )
    else:
        print(server)
        raise ValueError(f"{server} is not a valid Server name")
    return PATH


def WAV_PATH(data_obj, server):
    if server == "SWARA":
        prefix = os.path.join("/home", "mayank", "MTP", "begin_again")
    elif server == "AIRAWAT":
        prefix = os.path.join("/home", "mohith", "mayank")
    elif server == "DRISTI":
        prefix = os.path.join("/home", "aman", "mayank")
    elif server == "DHVANI":
        prefix = os.path.join("/home", "mayur", "mayank")
    else:
        raise ValueError(f"{server} is not a valid Server name")
    # if data_obj is Dataset.INDIC:
    #     PATH = os.path.join(
    #         "/home",
    #         "aman",
    #         "mayank",
    #     )
    # elif data_obj is Dataset.L2:
    #     PATH = os.path.join("")
    # elif data_obj is Dataset.MCV:
    #     PATH = os.path.join("mozilla", "cv-corpus-7.0-2021-07-21/en/wav/")
    # else:
    #     raise ValueError(f"{data_obj.name} is not a valid Dataset")

    return prefix


def BUDGET_TO_DURATION(budget):
    return floor(4.92 * budget)


def update_config(config):
    config["HOME_PATH"], config["DATASET_PATH"], config["WAV_PATH"] = get_path(
        Dataset[config["dataset"]], server=config["server"]
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

    print(get_path(Dataset["INDIC"], "SWARA"))
    print(get_path(Dataset["INDIC"], "DRISTI"))
    print(get_path(Dataset["INDIC"], "AIRAWAT"))
    print(get_path(Dataset["INDIC"], "DHVANI"))

    print(get_path(Dataset["L2"], "SWARA"))
    print(get_path(Dataset["L2"], "DRISTI"))
    print(get_path(Dataset["L2"], "AIRAWAT"))
    print(get_path(Dataset["L2"], "DHVANI"))
    
    print(get_path(Dataset["MCV"], "SWARA"))
    print(get_path(Dataset["MCV"], "DRISTI"))
    print(get_path(Dataset["MCV"], "AIRAWAT"))
    print(get_path(Dataset["MCV"], "DHVANI"))

    # print(get_data_path(Dataset['CORAAL'])) # Will raise an error.

    print(all_datasets())
