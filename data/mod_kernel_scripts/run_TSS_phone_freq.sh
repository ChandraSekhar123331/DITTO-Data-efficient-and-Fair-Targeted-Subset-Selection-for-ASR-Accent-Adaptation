# declare -a functions=('GCMI' 'FL2MI' 'FL1MI' 'LogDMI')
declare -a functions=('FL2MI')

curr_file_name=$(basename "$0")

echo $curr_file_name

mkdir -p logs/$curr_file_name/

echo "logging",  $curr_file_name

echo 
echo 

target=20
decay=$1

for fxn in "${functions[@]}"
do

    echo $curr_file_name $target $fxn 
    python TSS_phone_freq.py \
    --target $target \
    --fxn $fxn \
    --phoneme_decay $decay

done &> logs/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"