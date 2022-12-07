homepath=$(pwd)
finetunepath=$(cd ../models/quartznet_asr; pwd)
# acc/all/b_100/random/run_1/train.json
budget=$2
accent=$1

if [ ! -d random/$budget/run_1/ ]; then
  echo reached
  python select-random.py  --budget $budget --file_dir /home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/data/random/all_selection.json
fi
# "This is global random"  -- Chandra
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
   . scripts/finetune.sh $accent $data_dir
   echo "---------------beginning testing----------------------------"
   . scripts/test.sh $accent $data_dir
   cd "$homepath"
done
