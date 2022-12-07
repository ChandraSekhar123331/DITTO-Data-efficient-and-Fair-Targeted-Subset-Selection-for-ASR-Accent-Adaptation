homepath=$(pwd)
finetunepath=$(cd ../../models/quartznet_asr; pwd)
target=$4
budget=$3
feature=$2
speaker=$1
similarity="euclidean"
eta=1.0
declare -a fxns=('FL2MI' 'GCMI' 'LogDMI')
for fxn in "${fxns[@]}"
do
    echo
    echo "-----------------------tss----------------------------------"
    echo "$fxn"
    echo "$target" "$budget" "$similarity" "$eta" "$speaker" "$fxn" "$feature"
    echo
   python TSS-lite.py --target "$target" --budget "$budget" --similarity "$similarity" --eta "$eta" --speaker "$speaker" --fxn "$fxn" --feature_type "$feature" &> ./logs/selection-log-"$speaker"-"$feature"-"$budget"-"$target".txt
done
