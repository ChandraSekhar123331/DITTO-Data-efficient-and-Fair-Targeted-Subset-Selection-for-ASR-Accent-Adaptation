# json file containing pretrained_wers and pretrained_cers is assumed to be present
# the path to it is given as input by the argument = $file_name == $3
# Then need to do selection of top- b2 ones and also dump in the same folder.
# $b2 $file_dir $file_dir/train_wers_appended.json $expt_details
expt_details=$4
file_name=$3
file_dir=$2
budget=$1


echo $expt_details
echo $file_name


for run in 1 2 3
do
 echo doing run = $run of True WER
 python3 -u true_wer.py \
 --inp_json=$file_name \
 --budget=$budget \
 --exp_id=$run
done
