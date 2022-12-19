DATA=$(cd ../../mz-isca/expts/; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
accent=$1
rel_file_dir=$2
# IN=$rel_file_dir
# arrIN=(${IN//data/ })
# expt_details=${arrIN[1]}  

model_dir=$PRETRAINED_CKPTS/quartznet/finetuned/"$rel_file_dir"
echo making $model_dir before finetuning
echo train manifest is $DATA/$rel_file_dir/train.json
mkdir -p $model_dir
python3 -u finetune.py \
  --batch_size=16 \
  --num_epochs=100 \
  --eval_freq=1 \
  --train_freq=30 \
  --lr=1e-5 \
  --wav_dir="" \
  --train_manifest=$DATA/$rel_file_dir/train.json \
  --val_manifest=$DATA/$accent/dev.json \
  --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
  --output_dir=$model_dir/recent \
  --best_dir=$model_dir/best \
  --early_stop_patience=10 \
  --zero_infinity \
  --save_after_each_epoch \
  --turn_bn_eval \
  --ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
  --lr_decay=warmup \
  --seed=42 \
  --optimizer=novograd \
 > $model_dir/train_log.txt
echo