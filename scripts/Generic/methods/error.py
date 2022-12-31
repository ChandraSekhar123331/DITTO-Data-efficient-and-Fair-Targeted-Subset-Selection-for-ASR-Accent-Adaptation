import argparse
import os
import shutil
from pprint import pprint

from utils.dataset import all_datasets, all_servers, update_config
from utils.finetune import finetune
from utils.test import test


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=all_datasets(), type=str, required=True)
    parser.add_argument("--server", choices=all_servers(), type=str, required=True)
    parser.add_argument("--cuda", type=int, required=True, choices=[0, 1, 2, 3])
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--infer", action="store_true", default=False)
    parser.add_argument("--sample", action="store_true", default=False)
    parser.add_argument("--budget", type=int, required=True)

    parser.add_argument("--accent", required=True, type=str)
    parser.add_argument("--accent_dir_path", type=str, default="")

    parser.add_argument("--mode", type=str, choices=["seed_dev", "top"])

    parser.add_argument("--train_budget", type=int, choices=[100, 200], required=True)
    parser.add_argument("--pseudoTrans", action="store_true", default=False)

    parser.add_argument("--infer_json_path", default="", type=str)
    parser.add_argument("--infer_json_name", type=str, required=True)

    parser.add_argument("--finetune", action="store_true", default=False)
    parser.add_argument("--finetune_accent", default="", type=str)

    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--test_accent", default="", type=str)

    args = parser.parse_args()

    config = vars(args)
    if config["test"] and not config["test_accent"]:
        raise ValueError("test_accent can't be empty when --test switch is used")

    if config["finetune"] and not config["finetune_accent"]:
        raise ValueError(
            "finetune_accent can't be empty when --finetune switch is used"
        )

    return update_config(config)


def train_error(config):
    if not config["train"]:
        return
    # TODO: NEED to change the output_dir

    print("Training error model")
    SCRIPT_DIR = os.path.join(config["HOME_PATH"], "models", "error_model")
    PRETRAINED_CKPTS = os.path.join(
        config["HOME_PATH"], "models", "pretrained_checkpoints"
    )
    if config["mode"] == "top":
        assert config["INFER_JSON_NAME"] == "train.json"
        SETTING_PATH = os.path.join(
            config["accent_dir_path"],
            config["accent"],
            config["INFER_JSON_PATH"],
        )

        for run in range(1, 4):
            CKPT_PATH = os.path.join(
                PRETRAINED_CKPTS,
                "error_models",
                config["dataset"],
                SETTING_PATH,
                "trainBudget_{}".format(config["train_budget"]),
                "mode_{}".format(config["mode"]),
                f'pseudoTrans_{config["pseudoTrans"]}',
                f"run_{run}",
            )
            TRAIN_JSON_PATH = os.path.join(
                config["FULL_DATASET_PATH"],
                SETTING_PATH,
                f"run_{run}",
                config["INFER_JSON_NAME"],
            )
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
                --train_json_path={TRAIN_JSON_PATH} \
                {"--pseudoTrans" if config["pseudoTrans"] else ""} \
                --train_budget={config["train_budget"]} \
                --lr_decay=warmup \
                --seed={run} \
                --output_dir={os.path.join(CKPT_PATH, 'recent')} \
                --best_dir={os.path.join(CKPT_PATH, 'best')} \
                --pretrained_ckpt={os.path.join(PRETRAINED_CKPTS, "error_models", "librispeech", f"seed_{run}", "best", "ErrorClassifierPhoneBiLSTM_V2.pt")} \
                > {os.path.join(CKPT_PATH, "train_log.txt")}; 
                echo done
            """
            os.system(command=command)

    else:
        assert config["INFER_JSON_NAME"] == "train.json"
        assert False


def infer_error(config):
    if not config["infer"]:
        return

    print("Inferring error model")
    SCRIPT_DIR = os.path.join(config["HOME_PATH"], "models", "error_model")
    PRETRAINED_CKPTS = os.path.join(
        config["HOME_PATH"], "models", "pretrained_checkpoints"
    )
    SETTING_PATH = os.path.join(
        config["accent_dir_path"],
        config["accent"],
        config["INFER_JSON_PATH"],
    )

    if config["mode"] == "top":
        for run in range(1, 4):
            CKPT_PATH = os.path.join(
                PRETRAINED_CKPTS,
                "error_models",
                config["dataset"],
                SETTING_PATH,
                "trainBudget_{}".format(config["train_budget"]),
                "mode_{}".format(config["mode"]),
                f'pseudoTrans_{config["pseudoTrans"]}',
                f"run_{run}",
            )
            JSON_PATH = os.path.join(
                config["FULL_DATASET_PATH"],
                SETTING_PATH,
                f"run_{run}",
                config["INFER_JSON_NAME"],
            )
            OUTPUT_DIR = CKPT_PATH
            print("JSON_PATH is: ", JSON_PATH)
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
                --pretrained_ckpt={os.path.join(CKPT_PATH, "best", "ErrorClassifierPhoneBiLSTM_V2.pt")} \
                --output_dir={os.path.join(OUTPUT_DIR, "weights")} \
                {"--pseudoTrans" if config["pseudoTrans"] else ""} \
                > {os.path.join(OUTPUT_DIR, "infer_log.txt")}; 
                echo done
            """
            os.system(command=command)
    else:
        assert False


def sample_error(config):

    if not config["sample"]:
        return

    print("Sampling error model")
    SCRIPT_DIR = os.path.join(config["HOME_PATH"], "models", "error_model")
    PRETRAINED_CKPTS = os.path.join(
        config["HOME_PATH"], "models", "pretrained_checkpoints"
    )
    SETTING_PATH = os.path.join(
        config["accent_dir_path"],
        config["accent"],
        config["INFER_JSON_PATH"],
    )

    if config["mode"] == "top":
        for run in range(1, 4):
            JSON_PATH = os.path.join(
                config["FULL_DATASET_PATH"],
                SETTING_PATH,
                f"run_{run}",
                config["INFER_JSON_NAME"],
            )
            CKPT_PATH = os.path.join(
                PRETRAINED_CKPTS,
                "error_models",
                config["dataset"],
                SETTING_PATH,
                "trainBudget_{}".format(config["train_budget"]),
                "mode_{}".format(config["mode"]),
                f'pseudoTrans_{config["pseudoTrans"]}',
                f"run_{run}",
            )
            OUTPUT_DIR = os.path.join(
                config["FULL_DATASET_PATH"],
                SETTING_PATH,
                "budget_{}".format(config["budget"]),
                "error_model",
                "trainBudget_{}".format(config["train_budget"]),
                "mode_{}".format(config["mode"]),
                f'pseudoTrans_{config["pseudoTrans"]}',
            )
            LOG_PATH = os.path.join(
                CKPT_PATH, "budget_{}".format(config["budget"]), f"run_{run}"
            )
            os.makedirs(LOG_PATH, exist_ok=True)
            command = f"""
                cd {SCRIPT_DIR};
                DATA={config['FULL_DATASET_PATH']};
                PRETRAINED_CKPTS={PRETRAINED_CKPTS};
                accent={config['accent']};
                run={run};
                echo $accent run $run JSON {JSON_PATH};
                CUDA_VISIBLE_DEVICES={config['cuda']} \
                python -u error_model_sampling.py \
                --selection_json_file={JSON_PATH} \
                --seed_json_file={JSON_PATH} \
                --error_model_weights={os.path.join(CKPT_PATH, "weights", "weights.pkl")} \
                --exp_id=$run \
                --budget={config["budget"]} \
                --seed_budget={config["train_budget"]} \
                --selection_skip_cnt={config["train_budget"]} \
                --output_dir={OUTPUT_DIR} \
                > {os.path.join(LOG_PATH, "sampling_log.txt")}; 
                echo done
            """
            os.system(command=command)

    else:
        assert False


def finetune_error(config):
    if not config["finetune"]:
        return

    print("Finetuning Error model")

    PRETRAINED_CKPTS = os.path.join(
        config["HOME_PATH"], "models", "pretrained_checkpoints"
    )

    if config["mode"] == "top":
        for run in range(1, 4):
            OLD_SETTING_PATH = os.path.join(
                config["accent_dir_path"],
                config["accent"],
                config["INFER_JSON_PATH"],
                f"budget_{config['budget']}",
                "error_model",
                "trainBudget_{}".format(config["train_budget"]),
                "mode_{}".format(config["mode"]),
                f'pseudoTrans_{config["pseudoTrans"]}',
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
                config["accent_dir_path"],
                config["accent"],
                os.path.sep.join(config["INFER_JSON_PATH"].split(os.path.sep)[1:]),
                f"budget_{config['budget']}",
                "error_model",
                "trainBudget_{}".format(config["train_budget"]),
                "mode_{}".format(config["mode"]),
                f'pseudoTrans_{config["pseudoTrans"]}',
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
    else:
        assert False


def test_error(config):
    if not config["test"]:
        return
    print("Testing Error model")
    PRETRAINED_CKPTS = os.path.join(
        config["HOME_PATH"], "models", "pretrained_checkpoints"
    )
    if config["mode"] == "top":
        for run in range(1, 4):
            SETTING_PATH = os.path.join(
                config["test_accent"],
                "results",
                f"budget_{config['budget']}",
                config["accent_dir_path"],
                config["accent"],
                os.path.sep.join(config["INFER_JSON_PATH"].split(os.path.sep)[1:]),
                f"budget_{config['budget']}",
                "error_model",
                "trainBudget_{}".format(config["train_budget"]),
                "mode_{}".format(config["mode"]),
                f'pseudoTrans_{config["pseudoTrans"]}',
                f"run_{run}",
            )
            DUMP_PATH = os.path.join(config["FULL_DATASET_PATH"], SETTING_PATH)
            print(DUMP_PATH)
            CKPT_PATH = os.path.join(
                PRETRAINED_CKPTS,
                "quartznet",
                "finetuned",
                config["dataset"],
                SETTING_PATH,
            )
            TEST_JSON_PATH = os.path.join(
                config["FULL_DATASET_PATH"], config["test_accent"], "test.json"
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


if __name__ == "__main__":
    config = get_args()
    config["INFER_JSON_PATH"] = config["infer_json_path"]
    config["INFER_JSON_NAME"] = config["infer_json_name"]
    pprint(config)

    train_error(config)
    infer_error(config)
    sample_error(config)
    finetune_error(config)
    test_error(config)
