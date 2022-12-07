ERROR_MODEL_SCRIPTS=$(cd ../../models/error_model/; pwd)
PRETRAINED_CKPTS=$(cd ../../models/pretrained_checkpoints; pwd)
DATA=$(cd /home/mayank/MTP/begin_again/Error-Driven-ASR-Personalization/mz-isca/expts/; pwd)
# $accent $b2 $file_dir $file_dir/train.json $expt_details

file_name=$4
file_dir=$3
budget=$2
accent=$1

IN=$file_dir
arrIN=(${IN//_english/ })
expt_details=${arrIN[1]}  
echo $expt_details
echo $file_name

for run in 1 2 3
do
 mkdir -p $PRETRAINED_CKPTS/error_models/"$accent"/run_"$run"/
 python3 -u $ERROR_MODEL_SCRIPTS/infer_error_model.py \
      --batch_size=64 \
      --num_layers=4 \
      --hidden_size=64 \
      --input_size=64 \
      --seed=$run \
      --json_path=$file_name \
      --pretrained_ckpt=$PRETRAINED_CKPTS/error_models/$accent/run_"$run"/best/ErrorClassifierPhoneBiLSTM_V2.pt \
      --output_dir=$PRETRAINED_CKPTS/error_models/"$accent""$expt_details"/run_"$run"/best \
    > $PRETRAINED_CKPTS/error_models/"$accent""$expt_details"/run_"$run"/infer_log.txt
done

echo error model processing complete, sample weights saved

for run in 1 2 3
do
 echo ""
 echo run $run of error model selection
 python3 -u $ERROR_MODEL_SCRIPTS/error_model_sampling.py \
      --selection_json_file=$file_name \
      --seed_json_file=$DATA/$accent/seed.json \
      --error_model_weights=$PRETRAINED_CKPTS/error_models/"$accent""$expt_details"/run_"$run"/best/weights.pkl \
      --exp_id=$run \
      --budget=$budget
done