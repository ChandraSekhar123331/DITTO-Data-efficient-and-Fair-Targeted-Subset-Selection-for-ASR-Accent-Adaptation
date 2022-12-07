# This script absolutely needs `error` environment to run
accent=$1
method=$2
top_k_percentile=$3
BASE_PATH=/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization
DATA=$BASE_PATH/data
PRETRAINED_CKPTS=$BASE_PATH/models/pretrained_checkpoints/
python_file=$BASE_PATH/models/quartznet_asr/inference.py
logits_file=$BASE_PATH/entropy-testing/logits-dump-$accent-$method-$top_k_percentile.file
ckpt_path=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt
toml_path=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml
train_json=$BASE_PATH/mz-isca/expts/$accent/manifests/all.json
echo $train_json
wav_dir=$BASE_PATH/mozilla/cv-corpus-7.0-2021-07-21/en/wav
output_file=$BASE_PATH/entropy-testing/output-file-$accent-$method-$top_k_percentile.txt
# val_manifest indicates the path to the json file of
# output_file will have the wer, cer, predicted transcipt, true transcript.
# order of sentences in output_file will be same as that in json
# ckpt needs to be changed according to need I think. Rightnow just pretrained quartznet.

temp_file=$BASE_PATH/entropy-testing/output-$accent-$method-$top_k_percentile.aux
python3 -u $python_file \
--batch_size=64 \
--output_file=$output_file \
--wav_dir=$wav_dir \
--val_manifest=$train_json \
--model_toml=$toml_path \
--ckpt=$ckpt_path \
--logits_save_to=$logits_file \
> $temp_file



# selected_json is the file in which the final entropy-selected output will appear.
python3 -u check_corrln.py \
--logits_file=$logits_file \
--wers_file=$output_file \
--method=$method \
--top_k_percentile=$top_k_percentile \


# rm $temp_file
rm $logits_file

