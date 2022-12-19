pwd
DATA=$(cd ../../mz-isca/expts/; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd)

accent=$1

for run in 1 2 3
do
LR=3e-4
echo $accent run $run
mkdir -p $PRETRAINED_CKPTS/error_models/"$accent"/orig_transc/run_"$run"/
python3 -u train_error_model_orig_transc.py \
  --batch_size=1 \
  --num_epochs=200 \
  --train_freq=20 \
  --lr=$LR \
  --num_layers=4 \
  --hidden_size=64 \
  --input_size=64 \
  --weight_decay=0.001 \
  --train_portion=0.65 \
  --hypotheses_path=$DATA/$accent/all/quartznet_outputs/seed_plus_dev_out.txt \
  --lr_decay=warmup \
  --seed=$run \
  --output_dir=$PRETRAINED_CKPTS/error_models/$accent/orig_transc/run_"$run"/recent \
  --best_dir=$PRETRAINED_CKPTS/error_models/$accent/orig_transc/run_"$run"/best \
  --pretrained_ckpt=$PRETRAINED_CKPTS/error_models/librispeech/seed_"$run"/best/ErrorClassifierPhoneBiLSTM_V2.pt \
  > $PRETRAINED_CKPTS/error_models/$accent/orig_transc/run_"$run"/train_log.txt
echo
done
# done
