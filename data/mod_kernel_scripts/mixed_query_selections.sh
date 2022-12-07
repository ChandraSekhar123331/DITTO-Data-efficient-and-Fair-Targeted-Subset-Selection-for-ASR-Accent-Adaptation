curr_file_name=$(basename "$0")

echo $curr_file_name

mkdir -p logs/$curr_file_name/

echo "logging",  $curr_file_name




target=$1
fxn=$2
budget=$3



echo target = $target and submod-fxn = $fxn

python TSS_mixed_query.py --target $target --fxn $fxn --budget $budget &> logs/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"