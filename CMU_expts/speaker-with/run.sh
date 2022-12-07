homepath=$(pwd)
finetunepath=$(cd ../../models/quartznet_asr; pwd)
target=10
budget=200
similarity="euclidean"
eta=1.0
feature="39"
speaker=$1
declare -a fxns=('FL2MI' 'GCMI' 'LogDMI')
for fxn in "${fxns[@]}"
do
    echo
    echo "---------------------------------------------------------"
    echo "$fxn"
    echo "$target" "$budget" "$similarity" "$eta" "$speaker" "$fxn" "$feature"
    echo
#    python TSS.py --target "$target" --budget "$budget" --similarity "$similarity" --eta "$eta" --speaker "$speaker" --fxn "$fxn" --feature_type "$feature"
    cd "$finetunepath"
    . l2_speaker-W_scripts/asr_finetune.sh
    . l2_speaker-W_scripts/asr_test.sh
    cd "$homepath"
done
