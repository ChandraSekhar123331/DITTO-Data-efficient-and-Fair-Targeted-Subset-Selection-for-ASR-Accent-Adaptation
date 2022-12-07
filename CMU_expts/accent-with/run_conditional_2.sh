homepath=$(pwd)
finetunepath=$(cd ../../models/quartznet_asr; pwd)
target=10
budget=100
similarity="euclidean"
eta=1.0
feature="39"
accents=$1
other_accent=$2
declare -a fxns=('FLMI' 'LogDMI')
for fxn in "${fxns[@]}"
do
    echo
    echo "---------------------------------------------------------"
    echo "$fxn"
    echo "$target" "$budget" "$similarity" "$eta" "$accents" "$fxn" "$feature"
    echo
#    python TSS_conditional.py --target "$target" --budget "$budget" --similarity "$similarity" --eta "$eta" --accent "$accents" --other_accent "$other_accent" --fxn "$fxn" --feature_type "$feature"
    cd "$finetunepath"
    . l2_accent_scripts/asr_finetune_conditional.sh
    . l2_accent_scripts/asr_test_conditional.sh
    cd "$homepath"
done
