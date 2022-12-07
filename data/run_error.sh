homepath=$(pwd)
finetunepath=$(cd ../models/quartznet_asr; pwd)


eta=1.0
similarity="euclidean"
domain_budget=$6
target=$5
budget=$3
feature=$2
accent=$1

declare -a fxns=('FL2MI' 'GCMI' 'LogDMI')
for fxn in "${fxns[@]}"
do
    echo
    echo "-----------------------tss----------------------------------"
    echo "$fxn"
    echo "$target" "$domain_budget" "$similarity" "$eta" "$accent" "$fxn" "$feature"
    echo
   python TSS.py --target "$target" --budget "$budget" --similarity "$similarity" --eta "$eta" --accent "$accent" --fxn "$fxn" --feature_type "$feature"
   echo "---------------beginning finetuning----------------------------"
   cd "$finetunepath"
   . scripts/asr_finetune.sh $accent
   echo "---------------beginning testing----------------------------"
   . scripts/asr_test.sh $accent
   cd "$homepath"
done