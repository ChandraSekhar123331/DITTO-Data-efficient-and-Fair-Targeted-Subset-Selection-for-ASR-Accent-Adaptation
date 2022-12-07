homepath=$(pwd)
finetunepath=$(cd ../../models/quartznet_asr; pwd)
target=$4
budget=$3
feature=$2
speaker=$1
echo
echo "-----------------------within----------------------------------"
echo "$target" "$budget" "$speaker" "$feature"
echo
# python TSS_within_equal_random.py --target "$target" --budget "$budget" --speaker "$speaker" --feature_type "$feature"
echo "---------------beginning finetuning----------------------------"
cd "$finetunepath"
. l2_speaker-WO_scripts/asr_finetune_within_equal_random.sh >> "$homepath"/logs/within-log-"$speaker"-"$feature"-"$budget"-"$target".txt 2>&1
echo "---------------beginning testing----------------------------"
. l2_speaker-WO_scripts/asr_test_within_equal_random.sh >> "$homepath"/logs/within-log-"$speaker"-"$feature"-"$budget"-"$target".txt 2>&1
cd "$homepath"