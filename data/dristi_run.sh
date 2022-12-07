homepath=$(pwd)
finetunepath=$(cd ../models/quartznet_asr; pwd)
target=10
budget=100
similarity="euclidean"
eta=1.0
feature="39"
accents=$1
declare -a fxns=('FL1MI' 'FL2MI' 'GCMI' 'LogDMI')
for fxn in "${fxns[@]}"
do
    echo
    echo "---------------------------------------------------------"
    echo "$fxn"
    echo "$target" "$budget" "$similarity" "$eta" "$accents" "$fxn" "$feature"
    echo
#    python TSS_w2v2.py --target "$target" --budget "$budget" --similarity "$similarity" --eta "$eta" --accent "$accents" --fxn "$fxn" --feature_type "$feature"
    cd "$finetunepath"
    . scripts/dristi_asr_finetune.sh
    . scripts/dristi_asr_test.sh
    cd "$homepath"
done
