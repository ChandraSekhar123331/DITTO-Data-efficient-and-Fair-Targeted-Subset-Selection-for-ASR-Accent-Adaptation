DATA=$(cd ../../data; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
for seed in 1 2 3
do 
  for size in "$budget"
  do
    for accent in "${accents[@]}"
    do
      echo $accent $seed $size
      echo 
      echo
      model_dir=$PRETRAINED_CKPTS/quartznet/finetuned/$accent/$size/seed_"$seed"/error_model
      mkdir -p $model_dir
      python3 -u finetune.py \
        --batch_size=16 \
        --num_epochs=100 \
        --eval_freq=1 \
        --train_freq=30 \
        --lr=1e-5 \
        --wav_dir=$DATA/indicTTS_audio/indicTTS/$accent/english/wav \
        --train_manifest=$DATA/$accent/manifests/train/error_model/$size/seed_"$seed"/train.json \
        --val_manifest=$DATA/$accent/manifests/dev.json \
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
    done
    rm $DATA/$accent/manifests/train/error_model/$size/seed_"$seed"/train.json 
  done
done
