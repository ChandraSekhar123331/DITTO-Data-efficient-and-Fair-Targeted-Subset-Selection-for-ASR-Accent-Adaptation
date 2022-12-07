homepath=$(pwd)
finetunepath=$(cd ../../models/quartznet_asr; pwd)
eta=1.0
similarity="euclidean"
target=$4
budget=$3
feature=$2
accent=$1
# declare -a fxns=('FL2MI' 'GCMI' 'LogDMI')
declare -a fxns=('GCMI')
for fxn in "${fxns[@]}"
do
    echo
#     echo "------------------------TSS---------------------------------"
    echo "$fxn"
    echo "$target" "$budget" "$similarity" "$eta" "$accent" "$fxn" "$feature"
    echo
#    python TSS.py --target "$target" --budget "$budget" --similarity "$similarity" --eta "$eta" --accent "$accent" --fxn "$fxn" --feature_type "$feature"
    cd "$finetunepath"
    . mz_accent_scripts/asr_finetune.sh >> "$homepath"/logs/tss-log-"$accent"-"$feature"-"$budget"-"$target".txt 2>&1
    . mz_accent_scripts/asr_test.sh >> "$homepath"/logs/tss-log-"$accent"-"$feature"-"$budget"-"$target".txt 2>&1
    cd "$homepath"
done
