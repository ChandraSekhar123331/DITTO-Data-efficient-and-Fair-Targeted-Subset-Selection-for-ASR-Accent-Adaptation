curr_file_name=$(basename "$0")

echo $curr_file_name

mkdir -p logs/$curr_file_name/

echo "logging",  $curr_file_name

echo 
echo 


eta=1.0
similarity="euclidean"
feature=$4
target=$3
budget=$2
accent=$1

# declare -a fxns=('FL2MI' 'GCMI' 'LogDMI')
declare -a fxns=('FL2MI')
for fxn in "${fxns[@]}"
do
    echo
    echo "-----------------------tss----------------------------------"
    echo "$fxn"
    echo "$target" "$budget" "$accent" "$fxn" "$feature"
    echo
   python TSS.py --target "$target" --budget "$budget" --similarity "$similarity" --eta "$eta" --accent "$accent" --fxn "$fxn" --feature_type "$feature"
done &> logs/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"