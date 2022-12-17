import argparse
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
    args = parser.parse_args()
    config = vars(args)
    return update_config(config)


def pretrain_out(config):
    SCRIPT_DIR = os.path.join(config["HOME_PATH"], "models", "quartznet_asr")
    PRETRAINED_CKPTS = os.path.join(
        config["HOME_PATH"], "models", "pretrained_checkpoints"
    )
    SETTING_PATH = os.path.join(config["accent"], config["INFER_JSON_PATH"])
    JSON_PATH = os.path.join(
        config["FULL_DATASET_PATH"], SETTING_PATH, config["INFER_JSON_NAME"]
    )
    assert config["INFER_JSON_NAME"][-5:] == ".json"
    JSON_BASE_NAME = config["INFER_JSON_NAME"][:-5]

    OUTPUT_PATH = os.path.join(
        config["FULL_DATASET_PATH"], SETTING_PATH, "quartznet_outputs"
    )
    command = f"""
        cd {SCRIPT_DIR};
        DATA={config["FULL_DATASET_PATH"]};
        PRETRAINED_CKPTS={PRETRAINED_CKPTS}; 
        accent={config["accent"]};
        output_dir={OUTPUT_PATH};
        mkdir -p $output_dir;
        echo $accent $output_dir;
        echo;
        CUDA_VISIBLE_DEVICES={config['cuda']} \
        python -u inference.py \
        --batch_size=64 \
        --output_file=$output_dir/{f"{JSON_BASE_NAME}_out.txt"} \
        --wav_dir={config["FULL_WAV_PATH"]} \
        --val_manifest={JSON_PATH} \
        --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
        --ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
        > $output_dir/{f"{JSON_BASE_NAME}_infer_log.txt"};
        echo done;
    """

    os.system(command=command)


if __name__ == "__main__":
    config = get_args()
    config["INFER_JSON_PATH"] = config["json_path"]
    config["INFER_JSON_NAME"] = config["json_name"]
    pprint(config)

    pretrain_out(config)
