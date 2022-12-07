curr_file_name=$(basename "$0")

echo $curr_file_name

mkdir -p logs/$curr_file_name/

echo "logging",  $curr_file_name

echo 
echo 







homepath=$(pwd)
finetunepath=$(cd ../../models/quartznet_asr; pwd)
# acc/all/b_100/random/run_1/train.json
budget=$2
accent=$1

if [ ! -d random/$budget/run_1/ ]; then
  echo reached
  python select-random.py  --budget $budget --file_dir /home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/mz-isca/expts/random/all_selection.json
fi

# This is global random
for run in 1 2 3
do
   echo
   echo "--------------------full random-----------------------"
   echo "$accent" "$budget"
   data_dir=$accent/all/budget_$budget/random/run_$run
   mkdir -p $data_dir
   cp random/$budget/run_$run/train.json $data_dir/train.json
   echo "---------------beginning finetuning----------------------------"
   cd "$finetunepath"
   . mz-isca-scripts/finetune.sh $accent $data_dir
   echo "---------------beginning testing----------------------------"
   . mz-isca-scripts/test.sh $accent $data_dir
   cd "$homepath"
done &> logs/$curr_file_name/"$(date '+%Y-%m-%d %H:%M:%S')"
