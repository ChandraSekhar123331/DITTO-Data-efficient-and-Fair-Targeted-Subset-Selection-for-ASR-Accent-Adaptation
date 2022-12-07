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
#python TSS_random.py --target "$target" --budget "$budget" --speaker "$speaker" --feature_type "$feature"
cd "$finetunepath"
. l2_speaker-WO_scripts/dristi_asr_finetune_random.sh
. l2_speaker-WO_scripts/dristi_asr_test_random.sh
cd "$homepath"
