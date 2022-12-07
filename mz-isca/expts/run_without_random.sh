homepath=$(pwd)
finetunepath=$(cd ../../models/quartznet_asr; pwd)
target=$4
budget=$3
feature=$2
accent=$1
echo
echo "-----------------------random without----------------------------------"
echo "$target" "$budget" "$accent" "$feature"
echo
# python TSS_without_random.py --target "$target" --budget "$budget" --accent "$accent" --feature_type "$feature"
echo "---------------beginning finetuning----------------------------"
cd "$finetunepath"
. mz_accent_scripts/asr_finetune_without_random.sh >> "$homepath"/logs/random_without-log-"$accent"-"$feature"-"$budget"-"$target".txt 2>&1
echo "---------------beginning testing----------------------------"
. mz_accent_scripts/asr_test_without_random.sh >> "$homepath"/logs/random_without-log-"$accent"-"$feature"-"$budget"-"$target".txt 2>&1
cd "$homepath"
