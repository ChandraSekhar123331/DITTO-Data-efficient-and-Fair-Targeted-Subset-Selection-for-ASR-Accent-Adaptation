homepath=$(pwd)
finetunepath=$(cd ../../models/quartznet_asr; pwd)
target=$4
budget=$3
feature=$2
speaker=$1
echo
echo "-----------------------random----------------------------------"
echo "$target" "$budget" "$speaker" "$feature"
echo
python TSS_random.py --target "$target" --budget "$budget" --speaker "$speaker" --feature_type "$feature"
echo "---------------beginning finetuning----------------------------"
cd "$finetunepath"
. l2_speaker-WO_scripts/asr_finetune_random.sh >> "$homepath"/logs/random-log-"$speaker"-"$feature"-"$budget"-"$target".txt 2>&1
echo "---------------beginning testing----------------------------"
. l2_speaker-WO_scripts/asr_test_random.sh >> "$homepath"/logs/random-log-"$speaker"-"$feature"-"$budget"-"$target".txt 2>&1
cd "$homepath"
