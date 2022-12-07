homepath=$(pwd)
finetunepath=$(cd ../../models/quartznet_asr; pwd)
target=50
budget=100
similarity="euclidean"
eta=1.0
feature="39"
speaker=$1
declare -a fxns=('FL1MI' 'FL2MI' 'GCMI' 'LogDMI')
for fxn in "${fxns[@]}"
do
    echo
    echo "---------------------------------------------------------"
    echo "$fxn"
    echo "$target" "$budget" "$similarity" "$eta" "$speaker" "$fxn" "$feature"
    echo
    python TSS.py --target "$target" --budget "$budget" --similarity "$similarity" --eta "$eta" --speaker "$speaker" --fxn "$fxn" --feature_type "$feature"
    cd $speaker/manifests/TSS_output/all/budget_"$budget"/target_"$target"/"$fxn"/eta_"$eta"/"$similarity"/"$feature"/
    sed -i 's/\/home\/mayank\/MTP\/begin_again/\/mnt\/data\/aman\/mayank\/MTP\/mount_points/' run_*/train/train.json
    cd "$finetunepath"
    . l2_speaker-WO_scripts/dristi_asr_finetune.sh
    . l2_speaker-WO_scripts/dristi_asr_test.sh
    cd "$homepath"
done
