homepath=$(pwd)
finetunepath=$(cd ../models/quartznet_asr; pwd)
target=20
budget=200
similarity="euclidean"
accents=$1
# declare -a fxns=('FL2MI' 'LogDMI' 'GCMI')
declare -a fxns=('LogD')
for fxn in "${fxns[@]}"
do
    echo
    echo "---------------------------------------------------------"
    echo "$fxn"
    echo "$target" "$budget" "$similarity" "$accents" "$fxn"
    echo
    python TSS_within.py --target "$target" --budget "$budget" --similarity "$similarity" --accent "$accents" --fxn "$fxn"
    cd "$finetunepath"
    . scripts/asr_finetune.sh
    . scripts/asr_test_within.sh
    cd "$homepath"
done
