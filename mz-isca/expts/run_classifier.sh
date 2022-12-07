homepath=$(pwd)
finetunepath=$(cd ../../models/quartznet_asr; pwd)
budget=$2
accent=$1
echo
echo "-----------------------classifier_based----------------------------------"
echo "$budget" "$accent"
echo
echo "---------------beginning finetuning----------------------------"
cd "$finetunepath"
. mz_accent_scripts/asr_finetune_classifier.sh >> "$homepath"/logs/classifier-log-"$accent"-"$budget".txt 2>&1
echo "---------------beginning testing----------------------------"
. mz_accent_scripts/asr_test_classifier.sh >> "$homepath"/logs/classifier-log-"$accent"-"$budget".txt 2>&1
cd "$homepath"
