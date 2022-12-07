ERROR_MODEL_SCRIPTS=$(cd ../entropy-testing/pseudo-transcript-entropy/models/error_model/; pwd)
DATA=$(cd /home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/mz-isca/expts/; pwd)
PRETRAINED_CKPTS=$(cd ../entropy-testing/pseudo-transcript-entropy/models/pretrained_checkpoints; pwd)
expt_details=$7
stage2_sim=$6
ngram=$5
file_name=$4
file_dir=$3
budget=$2
accent=$1
# echo $accent $b2 $file_dir $file_dir/train.json
 
echo $expt_details
echo $file_name
echo ngram = $ngram
echo stage2 similairty metric = $stage2_sim
for run in 1 2 3
do
 echo doing run = $run of tf-idf based diversity.
 mkdir -p $PRETRAINED_CKPTS/error_models/$file_dir/budget_$budget/div_tf_idf_"$ngram"gram_"$stage2_sim"/run_"$run"/
 python3 -u $ERROR_MODEL_SCRIPTS/select_tf_idf.py \
      --seed=$run \
      --selection_json_file=$file_name \
      --exp_id=$run \
      --budget=$budget \
      --ngram=$ngram \
      --metric=$stage2_sim \
    > $PRETRAINED_CKPTS/error_models/$file_dir/budget_$budget/div_tf_idf_"$ngram"gram_"$stage2_sim"/run_"$run"/infer_log.txt
done
