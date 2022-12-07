homepath=$(pwd)
finetunepath=$(cd ../models/quartznet_asr; pwd)
target=50
budget=100
similarity="euclidean"
accents=$1
declare -a fxns=('FL' 'LogD')
for fxn in "${fxns[@]}"
do
    echo
    echo "---------------------------------------------------------"
    echo "$fxn"
    echo "$target" "$budget" "$similarity" "$accents" "$fxn"
    echo
    python TSS_within.py --target "$target" --budget "$budget" --similarity "$similarity" --accent "$accents" --fxn "$fxn"
    cd $accents/manifests/train/error_model/"$budget"/
    sed -i 's/\/home\/mayank\/MTP\/begin_again/\/mnt\/data\/aman\/mayank\/MTP\/mount_points/' seed*/*json
    cd "$finetunepath"
    . scripts/dristi_asr_finetune.sh
    . scripts/dristi_asr_test_within.sh
    cd "$homepath"
done
