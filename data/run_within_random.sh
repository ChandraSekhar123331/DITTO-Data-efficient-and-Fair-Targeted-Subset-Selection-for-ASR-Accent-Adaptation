homepath=$(pwd)
finetunepath=$(cd ../models/quartznet_asr; pwd)
target=20
budget=200
feature="39"
accent=$1
echo
echo "---------------------------------------------------------"
echo "$target" "$budget" "$accent" "$feature"
echo
# python TSS_within_random.py --target "$target" --budget "$budget" --accent "$accent" --feature_type "$feature"
cd "$finetunepath"
. scripts/asr_finetune_within_random.sh $accent
. scripts/asr_test_within_random.sh $accent
cd "$homepath"
