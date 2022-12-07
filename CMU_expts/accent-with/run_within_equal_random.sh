homepath=$(pwd)
finetunepath=$(cd ../../models/quartznet_asr; pwd)
target=10
budget=100
feature="39"
accents=$1
echo
echo "---------------------------------------------------------"
echo "$target" "$budget" "$accents" "$feature"
echo
python TSS_within_equal_random.py --target "$target" --budget "$budget" --accent "$accents" --feature_type "$feature"
#cd "$finetunepath"
#. l2_accent_scripts/asr_finetune_within_equal_random.sh
#. l2_accent_scripts/asr_test_within_equal_random.sh
#cd "$homepath"
