import argparse
import json
import os
from pprint import pprint

from utils.dataset import all_datasets, update_config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=all_datasets(), type=str, required=True)
    parser.add_argument("--cuda", type=int, required=True, choices=[0, 1, 2, 3])
    parser.add_argument("--accent", type=str, required=True)
    parser.add_argument("--json_path", default="", type=str)
    parser.add_argument("--json_name", type=str, required=True)
    parser.add_argument("--output_json_name", type=str, required=True)
    args = parser.parse_args()
    return update_config(vars(args))


def add_json_attributes(config):
    SETTING_PATH = os.path.join(config["accent"], config["INFER_JSON_PATH"])
    JSON_PATH = os.path.join(
        config["FULL_DATASET_PATH"], SETTING_PATH, config["INFER_JSON_NAME"]
    )
    OUT_PATH = os.path.join(
        config["FULL_DATASET_PATH"], SETTING_PATH, config["OUTPUT_JSON_NAME"]
    )
    assert config["INFER_JSON_NAME"][-5:] == ".json"
    JSON_BASE_NAME = config["INFER_JSON_NAME"][:-5]

    LOG_PATH = os.path.join(
        config["FULL_DATASET_PATH"],
        SETTING_PATH,
        "quartznet_outputs",
        f"{JSON_BASE_NAME}_out.txt",
    )
    print(LOG_PATH)
    assert os.path.isfile(LOG_PATH)

    log_dict = {}
    with open(LOG_PATH) as log_file:
        lines = [line.strip() for line in log_file.readlines()]
        lines = [_ for _ in lines if len(_) > 0]

        assert len(lines) % 5 == 0
        for ind in range(0, len(lines), 5):
            path = lines[ind]
            WER = float(lines[ind + 1][len("WER: ") :].strip())
            CER = float(lines[ind + 2][len("CER: ") :].strip())
            REF = lines[ind + 3][len("Ref: ") :].strip()
            HYP = lines[ind + 4][len("Hyp: ") :].strip()
            log_dict[path] = {
                "WER": WER,
                "CER": CER,
                "REF": REF,
                "HYP": HYP,
            }
    with open(JSON_PATH) as json_file:
        json_lines = [json.loads(line.strip()) for line in json_file.readlines()]

    with open(OUT_PATH, "w") as out_file:
        for sample in json_lines:
            # print(sample["text"])
            path = sample["audio_filepath"]
            # print(log_dict[path]["REF"])
            sample["WER"] = log_dict[path]["WER"]
            sample["CER"] = log_dict[path]["CER"]
            sample["REF"] = log_dict[path]["REF"]
            sample["pseudo_text"] = log_dict[path]["HYP"]
            out_file.write(json.dumps(sample))
            out_file.write("\n")
            # try:
            #     assert(sample["pseudo_text"] == log_dict[path]["HYP"])
            # except:
            #     print(sample["pseudo_text"])
            #     print(log_dict[path]["HYP"])


if __name__ == "__main__":
    config = get_args()
    config["INFER_JSON_PATH"] = config["json_path"]
    config["INFER_JSON_NAME"] = config["json_name"]
    config["OUTPUT_JSON_NAME"] = config["output_json_name"]

    pprint(config)

    add_json_attributes(config)
