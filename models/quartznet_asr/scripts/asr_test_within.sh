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
      python3 -u inference.py \
      --batch_size=64 \
      --output_file=$model_dir/test_out.txt \
      --wav_dir=$DATA/indicTTS_audio/indicTTS/$accent/english/wav \
      --val_manifest=$DATA/$accent/manifests/test.json \
      --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
      --ckpt=$model_dir/best/Jasper.pt \
      > $model_dir/test_infer_log.txt
      output_dir=$DATA/$accent/manifests/TSS_output/within/budget_"$budget"/target_"$target"/"$fxn"/"$similarity"/run_"$seed"/output/
      cp $model_dir/test_infer_log.txt $output_dir
    done
  done
done
rm -r $PRETRAINED_CKPTS/quartznet/finetuned/$accents/$budget


