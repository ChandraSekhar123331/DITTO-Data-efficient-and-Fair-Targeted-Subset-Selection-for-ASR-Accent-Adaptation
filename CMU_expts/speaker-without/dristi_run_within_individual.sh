homepath=$(pwd)
finetunepath=$(cd ../../models/quartznet_asr; pwd)
target=10
budget=100
feature="39"
speaker="$1"
other_speaker="$2"

echo
echo "---------------------------------------------------------"
echo "$target" "$budget" "$speaker" "$feature" "$other_speaker"
echo
#python TSS_within_individual.py --target "$target" --budget "$budget" --speaker "$speaker" --feature_type "$feature" --other_speaker "$other_speaker"
cd "$finetunepath"
. l2_speaker-WO_scripts/dristi_asr_finetune_within_individual.sh
. l2_speaker-WO_scripts/dristi_asr_test_within_individual.sh
cd "$homepath"
