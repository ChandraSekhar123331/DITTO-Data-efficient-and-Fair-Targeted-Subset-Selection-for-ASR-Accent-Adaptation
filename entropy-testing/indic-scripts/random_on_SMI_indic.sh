accent=$1
budget=$2
target=$3
SMI_method=$4
eta=$5
features=$6
run=$7
random_budget=$8

echo generating entropy-samples for accent=$accent budget=$budget $SMI_method run=$run

base_path=/home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization
results_prefix_path=$base_path/data/$accent/manifests/TSS_output/all/budget_$budget/target_$target/$SMI_method/eta_$eta/euclidean/$features/run_$run
train_json=$results_prefix_path/train/train.json
echo train.json is $train_json
echo
echo Creating directory $results_prefix_path/random/
echo
mkdir -p $results_prefix_path/random/


# create the entropy-file
# This script absolutely needs `error` environment to run
# PRETRAINED_CKPTS=$base_path/models/pretrained_checkpoints/
# python_file=$base_path/models/quartznet_asr/inference.py
# logits_file=$base_path/entropy-testing/logits-dump.file
# ckpt_path=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt
# toml_path=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml
# output_file=$results_prefix_path/entropy/prediction-pretrained.txt
# val_manifest indicates the path to the json file of
# output_file will have the wer, cer, predicted transcipt, true transcript.
# order of sentences in output_file will be same as that in json
# ckpt needs to be changed according to need I think. Rightnow just pretrained quartznet.

# temp_file=$results_prefix_path/entropy/wer-cer-pretrained.txt
# python3 -u $python_file \
# --batch_size=8 \
# --output_file=$output_file \
# --wav_dir="" \
# --val_manifest=$train_json \
# --model_toml=$toml_path \
# --ckpt=$ckpt_path \
# --logits_save_to=$logits_file \
# > $temp_file




echo random-budget is $random_budget

shuf -n $random_budget $train_json > $results_prefix_path/random/selected-$random_budget.json

# python3 -u $base_path/entropy-testing/entropy.py \
# --train_json=$train_json \
# --selected_json=$results_prefix_path/entropy/selected-$entropy_budget.json \
# --selection_count=$random_budget

# echo removing $logits_file
# rm $logits_file

# echo removing $output_file
# rm $output_file