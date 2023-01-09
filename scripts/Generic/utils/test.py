import os

from utils.dataset import HOME_PATH


def test(
    DUMP_PATH, TEST_JSON_PATH, CKPT_PATH, PRETRAINED_CKPTS, WAV_PATH, CUDA, SERVER
):
    assert os.path.isdir(DUMP_PATH)
    HOME = HOME_PATH(server=SERVER)
    SCRIPT_DIR = os.path.join(HOME, "models", "quartznet_asr")
    MODEL_DIR = CKPT_PATH
    command = f"""
        cd {SCRIPT_DIR};
        model_dir={MODEL_DIR};
        PRETRAINED_CKPTS={PRETRAINED_CKPTS};
        CUDA_VISIBLE_DEVICES={CUDA} \
        python -u inference.py \
        --batch_size=64 \
        --output_file={os.path.join(MODEL_DIR, "test_out.txt")} \
        --wav_dir={WAV_PATH} \
        --val_manifest={TEST_JSON_PATH} \
        --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
        --ckpt=$model_dir/best/Jasper.pt \
        > $model_dir/test_infer_log.txt;
        cp -v $model_dir/test_infer_log.txt {DUMP_PATH};
    echo TESTING {TEST_JSON_PATH} completed;
    """

    os.system(command)


def dump_pretrain_logits(
    LOGITS_FILE, TEST_JSON_PATH, PRETRAINED_CKPTS, WAV_PATH, CUDA, SERVER
):
    assert os.path.isdir(os.path.dirname(LOGITS_FILE))
    HOME = HOME_PATH(server=SERVER)
    SCRIPT_DIR = os.path.join(HOME, "models", "quartznet_asr")
    command = f"""
        cd {SCRIPT_DIR};
        PRETRAINED_CKPTS={PRETRAINED_CKPTS};
        CUDA_VISIBLE_DEVICES={CUDA} \
        python -u inference.py \
        --batch_size=64 \
        --wav_dir={WAV_PATH} \
        --val_manifest={TEST_JSON_PATH} \
        --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
        --ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
        --logits_save_to={LOGITS_FILE} \
        > /dev/null; 

    echo Creating Pretrain Logits for {TEST_JSON_PATH} completed;
    """

    os.system(command)


if __name__ == "__main__":
    print("Check if test.py works!")
    print("!!!Not implemented yet!!!")
