homepath=$(pwd)
finetunepath=$(cd ../../models/quartznet_asr; pwd)
speaker=$1
echo
echo "---------------------------------------------------------"
echo "$target" "$budget" "$speaker" "$feature"
echo
python TSS_within_equal_random.py --target "$target" --budget "$budget" --speaker "$speaker" --feature_type "$feature"
#cd "$finetunepath"
#. l2_speaker-W_scripts/asr_finetune_within_equal_random.sh
#. l2_speaker-W_scripts/asr_test_within_equal_random.sh
#cd "$homepath"
