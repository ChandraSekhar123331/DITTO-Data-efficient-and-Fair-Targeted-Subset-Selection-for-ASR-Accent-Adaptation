homepath=$(pwd)
finetunepath=$(cd ../../models/quartznet_asr; pwd)
target=10
budget=100
feature="39"
speaker=$1
echo
echo "---------------------------------------------------------"
echo "$target" "$budget" "$speaker" "$feature"
echo
python TSS_within_random.py --target "$target" --budget "$budget" --speaker "$speaker" --feature_type "$feature"
cd "$finetunepath"
. l2_speaker-W_scripts/asr_finetune_within_random.sh
. l2_speaker-W_scripts/asr_test_within_random.sh
cd "$homepath"
