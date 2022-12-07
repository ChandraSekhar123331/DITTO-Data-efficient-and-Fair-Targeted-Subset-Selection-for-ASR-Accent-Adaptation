homepath=$(pwd)
finetunepath=$(cd ../models/quartznet_asr; pwd)
target=$4
budget=$3
feature=$2
accent=$1
echo
echo "-----------------------random----------------------------------"
echo "$target" "$budget" "$accent" "$feature"
echo
# python TSS_random.py --target "$target" --budget "$budget" --accent "$accent" --feature_type "$feature"
echo "---------------beginning finetuning----------------------------"
cd "$finetunepath"
. scripts/asr_finetune_random.sh "$accent"
echo "---------------beginning testing----------------------------"
. scripts/asr_test_random.sh "$accent"
cd "$homepath"
