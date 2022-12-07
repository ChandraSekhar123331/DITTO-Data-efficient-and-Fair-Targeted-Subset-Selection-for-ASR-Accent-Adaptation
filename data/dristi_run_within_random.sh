homepath=$(pwd)
finetunepath=$(cd ../models/quartznet_asr; pwd)
target=10
budget=100
feature="39"
accents=$1
echo
echo "---------------------------------------------------------"
echo "$target" "$budget" "$accents" "$feature"
echo
#python TSS_within_random.py --target "$target" --budget "$budget" --accent "$accents" --feature_type "$feature"
cd "$finetunepath"
. scripts/dristi_asr_finetune_within_random.sh
. scripts/dristi_asr_test_within_random.sh
cd "$homepath"
