import argparse
from pprint import pprint
from dataset import get_path, all_datasets, Dataset
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=all_datasets(), type=str, required=True)
    parser.add_argument("--cuda", type=int, required=True, choices=[0, 1, 2, 3])
    parser.add_argument("--pseudoTrans", action='store_true', default=False)
    parser.add_argument("--finetune", action='store_true', default=False)
    parser.add_argument("--test", action='store_true', default=False)
    parser.add_argument("--train", action='store_true', default=False)
    parser.add_argument("--infer", action='store_true', default=False)
    parser.add_argument("--accent", required=True, type=str)
    parser.add_argument("--json_path", default="", type=str)
    parser.add_argument("--json_name", type=str, required=True)
    args = parser.parse_args()    
    return vars(args)



def train_error(config):
    if not config["train"]: return 


    SCRIPT_DIR = os.path.join(config["HOME_PATH"], "models", "error_model")
    PRETRAINED_CKPTS=os.path.join(config["HOME_PATH"], "models", "pretrained_checkpoints")
    SETTING_PATH=os.path.join(config["accent"])
    for run in range(1, 4):
        CKPT_PATH=os.path.join(PRETRAINED_CKPTS, "error_models", config["dataset"], SETTING_PATH, f"run_{run}")
        command = f"""
            cd {SCRIPT_DIR};
            DATA={config['FULL_DATASET_PATH']};
            PRETRAINED_CKPTS={PRETRAINED_CKPTS};
            accent={config['accent']};
            LR=3e-4;
            run={run};
            echo $accent run $run;
            mkdir -pv {CKPT_PATH}; 
            CUDA_VISIBLE_DEVICES={config['cuda']} \
            python -u train_error_model.py \
            --batch_size=1 \
            --num_epochs=200 \
            --train_freq=20 \
            --lr=$LR \
            --num_layers=4 \
            --hidden_size=64 \
            --input_size=64 \
            --weight_decay=0.001 \
            --train_portion=0.65 \
            --hypotheses_path=$DATA/$accent/all/quartznet_outputs/seed_plus_dev_out.txt \
            --lr_decay=warmup \
            --seed={run} \
            --output_dir={os.path.join(CKPT_PATH, 'recent')} \
            --best_dir={os.path.join(CKPT_PATH, 'best')} \
            --pretrained_ckpt={os.path.join(PRETRAINED_CKPTS, "error_models", "librispeech", f"seed_{run}", "best", "ErrorClassifierPhoneBiLSTM_V2.pt")} \
            > {os.path.join(CKPT_PATH, "train_log.txt")}; 
            echo done
        """
        os.system(command=command)


def infer_error(config):
    if not config["infer"]: return

    SCRIPT_DIR = os.path.join(config["HOME_PATH"], "models", "error_model")
    PRETRAINED_CKPTS=os.path.join(config["HOME_PATH"], "models", "pretrained_checkpoints")
    SETTING_PATH=os.path.join(config["accent"], config["INFER_JSON_PATH"])
    for run in range(1, 4):
        JSON_PATH = os.path.join(config["FULL_DATASET_PATH"], SETTING_PATH, config["INFER_JSON_NAME"])
        print("JSON_PATH is: ", JSON_PATH)
        CKPT_PATH=os.path.join(PRETRAINED_CKPTS, "error_models", config["dataset"], SETTING_PATH, f"run_{run}")
        command = f"""
            cd {SCRIPT_DIR};
            DATA={config['FULL_DATASET_PATH']};
            PRETRAINED_CKPTS={PRETRAINED_CKPTS};
            accent={config['accent']};
            run={run};
            echo $accent run $run JSON {JSON_PATH};
            CUDA_VISIBLE_DEVICES={config['cuda']} \
            python -u infer_error_model.py \
            --batch_size=64 \
            --num_layers=4 \
            --hidden_size=64 \
            --input_size=64 \
            --json_path={JSON_PATH} \
            --pretrained_ckpt={os.path.join(PRETRAINED_CKPTS, "error_models", config['accent'], f"run_{run}", "best", "ErrorClassifierPhoneBiLSTM_V2.pt")} \
            --output_dir={os.path.join(CKPT_PATH), "weights"} \
            > {os.path.join(CKPT_PATH, "infer_log.txt")}; 
            echo done
        """
        os.system(command=command)


def sample_error(config):
    SCRIPT_DIR = os.path.join(config["HOME_PATH"], "models", "error_model")
    PRETRAINED_CKPTS=os.path.join(config["HOME_PATH"], "models", "pretrained_checkpoints")
    SETTING_PATH=os.path.join(config["accent"], config["INFER_JSON_PATH"])
    for run in range(1, 4):
        JSON_PATH = os.path.join(config["FULL_DATASET_PATH"], SETTING_PATH, config["INFER_JSON_NAME"])
        CKPT_PATH=os.path.join(PRETRAINED_CKPTS, "error_models", config["dataset"], SETTING_PATH, f"run_{run}")
        command = f"""
            cd {SCRIPT_DIR};
            DATA={config['FULL_DATASET_PATH']};
            PRETRAINED_CKPTS={PRETRAINED_CKPTS};
            accent={config['accent']};
            run={run};
            echo $accent run $run JSON {JSON_PATH};
            CUDA_VISIBLE_DEVICES={config['cuda']} \
            python -u error_model_sampling.py \
            --selection_json_file={config["JSON_PATH"]} \
            --seed_json_file=$DATA/$accent/seed.json \
            --error_model_weights={os.path.join(CKPT_PATH, "weights", "weights.pkl")} \
            --exp_id=$run \
            --budget={config["budget"]} \
            > {os.path.join(CKPT_PATH, "sampling_log.txt")}; 
            echo done
        """
        os.system(command=command)


def finetune_error(config):
    if not config["finetune"]: return


def test_error(config):
    if not config["test"]: return


if __name__ == "__main__":
    config = get_args()

    config["HOME_PATH"], config["DATASET_PATH"], config["WAV_PATH"] = get_path(Dataset[config["dataset"]])
    config["FULL_DATASET_PATH"] = os.path.join(config["HOME_PATH"], config["DATASET_PATH"])
    config["FULL_WAV_PATH"] = os.path.join(config["HOME_PATH"], config["WAV_PATH"])
    config["FULL_PATH"] = os.path.join(config["HOME_PATH"], config["DATASET_PATH"], config["accent"])
    config["INFER_JSON_PATH"] = config['json_path']
    config["INFER_JSON_NAME"] = config["json_name"]
    
    train_error(config)
    infer_error(config)
    finetune_error(config)
    test_error(config)
    