homepath=$(pwd)
finetunepath=$(cd ../../models/quartznet_asr; pwd)
target=50
budget=100
feature="39"
speaker=$1
echo
echo "---------------------------------------------------------"
echo "$target" "$budget" "$speaker" "$feature"
echo
python TSS_random.py --target "$target" --budget "$budget" --speaker "$speaker" --feature_type "$feature"
cd $speaker/manifests/TSS_output/all/budget_"$budget"/target_"$target"/"random"/
sed -i 's/\/home\/mayank\/MTP\/begin_again/\/mnt\/data\/aman\/mayank\/MTP\/mount_points/' run_*/train/train.json
cd "$finetunepath"
. l2_speaker-WO_scripts/dristi_asr_finetune_random.sh
. l2_speaker-WO_scripts/dristi_asr_test_random.sh
cd "$homepath"
