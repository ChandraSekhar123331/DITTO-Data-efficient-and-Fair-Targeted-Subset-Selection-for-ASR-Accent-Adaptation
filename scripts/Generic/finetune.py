import os
from dataset import HOME_PATH

def finetune(JSON_PATH, VAL_JSON_PATH, CKPT_PATH, PRETRAINED_CKPTS, WAV_PATH, CUDA):
    HOME = HOME_PATH()
    SCRIPT_DIR = os.path.join(HOME, "models", "quartznet_asr")
    MODEL_DIR = CKPT_PATH
    command = f"""
        cd {SCRIPT_DIR};
        model_dir={MODEL_DIR};
        PRETRAINED_CKPTS={PRETRAINED_CKPTS};
        echo making $model_dir before finetuning; 
        echo train manifest is {JSON_PATH};
        mkdir -p $model_dir
        CUDA_VISIBLE_DEVICES={CUDA} \
        python -u finetune.py \
        --batch_size=16 \
        --num_epochs=100 \
        --eval_freq=1 \
        --train_freq=30 \
        --lr=1e-5 \
        --wav_dir={WAV_PATH} \
        --train_manifest={JSON_PATH} \
        --val_manifest={VAL_JSON_PATH} \
        --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
        --output_dir={os.path.join(CKPT_PATH, "recent")} \
        --best_dir={os.path.join(CKPT_PATH, "best")} \
        --early_stop_patience=10 \
        --zero_infinity \
        --save_after_each_epoch \
        --turn_bn_eval \
        --ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
        --lr_decay=warmup \
        --seed=42 \
        --optimizer=novograd \
        > {os.path.join(CKPT_PATH, "train_log.txt")};
        echo done
    """

    os.system(command)



if __name__ == "__main__":
    print("Check if finetune.py works!")
    print("!!!Not implemented yet!!!")
