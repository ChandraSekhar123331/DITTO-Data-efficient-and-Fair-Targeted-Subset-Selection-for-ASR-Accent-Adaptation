UNIFORM_MODEL_SCRIPTS=$(cd ../entropy-testing/pseudo-transcript-entropy/models/error_model/; pwd)
DATA=$(cd /home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/data; pwd)
PRETRAINED_CKPTS=$(cd ../entropy-testing/pseudo-transcript-entropy/models/pretrained_checkpoints; pwd)


decay_factor=$5
file_name=$4
file_dir=$3
budget=$2
accent=$1
# echo $accent $b2 $file_dir $file_dir/train.json


IN=$file_dir
arrIN=(${IN//_english/ })
expt_details=${arrIN[1]}  
echo expt_details are $expt_details
echo file_name of stage1 is $file_name
for run in 1 2 3
do
 echo doing run = $run of Uniform phoneme based diversity.
 mkdir -p $PRETRAINED_CKPTS/error_models/$file_dir/budget_$budget/uniform_$decay_factor/run_"$run"/
 python3 -u $UNIFORM_MODEL_SCRIPTS/generate_uniform_samples.py \
      --selection_json_file=$file_name \
      --budget=$budget \
      --output_dir=$file_dir/budget_$budget/uniform_$decay_factor/run_"$run"/ \
      --decay_factor=$decay_factor \
    > $PRETRAINED_CKPTS/error_models/$file_dir/budget_$budget/uniform_$decay_factor/run_"$run"/output_log.txt
done
