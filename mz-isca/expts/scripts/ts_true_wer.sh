homepath=$(pwd)
finetunepath=$(cd ../../models/quartznet_asr; pwd)
budget=$2
accent=$1



echo
echo
echo "--------------------True WER based selections-----------------------"   
cd "$finetunepath"
expt_details=/all/within/
file_dir=$accent/all/within/
. mz-isca-scripts/chandra_pretrained_test.sh $accent $file_dir $expt_details

cd "$homepath"
pwd
python3 -u append_wers_to_json.py --file_dir $file_dir

. scripts/true_wer.sh $budget $file_dir $file_dir/train_wers_appended.json $expt_details
for run in 1 2 3
do
    file_dir=$accent/all/within/budget_$budget/true_wer/run_$run
    mkdir -pv $file_dir
    echo "---------------beginning finetuning----------------------------"
    cd "$finetunepath"
    . mz-isca-scripts/finetune.sh $accent $file_dir
    echo "---------------beginning testing----------------------------"
    . mz-isca-scripts/test.sh $accent $file_dir
    cd "$homepath"
done
