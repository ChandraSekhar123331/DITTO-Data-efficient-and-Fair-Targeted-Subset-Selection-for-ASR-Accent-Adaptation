curr_file_name=$(basename "$0")

echo $curr_file_name

mkdir -p logs/$curr_file_name/

echo "logging",  $curr_file_name

echo 
echo 







homepath=$(pwd)
finetunepath=$(cd ../../models/quartznet_asr; pwd)
# acc/all/b_100/t_10/GCMI/39/b_50/top/run_1/train.json

budget=$2
accent=$1

# . scripts/tss.sh $accent $b1 $tar $ft
# declare -a fxns=('FL2MI' 'GCMI' 'LogDMI')
declare -a fxns=('FL2MI')
for fxn in "${fxns[@]}"
do
   echo
   echo
   echo "-------------------- within random-selection-----------------------"
   echo "$accent" "$budget"
   

   mkdir -pv $accent/all/within/
   cp -v $accent/selection.json $accent/all/within/train.json
   
   file_dir=$accent/all/within/
   . scripts/rand.sh $budget $file_dir
   for run in 1 2 3
   do
       file_dir=$accent/all/within/budget_$budget/random/run_$run
       echo "---------------beginning finetuning----------------------------"
       cd "$finetunepath"
       mkdir -pv $file_dir
       . mz-isca-scripts/finetune.sh $accent $file_dir
       echo "---------------beginning testing----------------------------"
       . mz-isca-scripts/test.sh $accent $file_dir
       cd "$homepath"
   done
done &> logs/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"
