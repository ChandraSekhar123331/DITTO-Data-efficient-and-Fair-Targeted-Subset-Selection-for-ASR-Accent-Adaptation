DATA=$(
  cd ../../data
  pwd
)
PRETRAINED_CKPTS=$(
  cd ../pretrained_checkpoints
  pwd
)

declare -a accents=('kannada_male_english' 'rajasthani_male_english' 'gujarati_female_english' 'hindi_male_english' 'malayalam_male_english' 'assamese_female_english' 'manipuri_female_english' 'tamil_male_english')

for accent in "${accents[@]}"; do
  for run in 1 2 3; do
    LR=3e-4
    echo $accent run $run
    python3 -u train_error_model.py \
      --batch_size=1 \
      --num_epochs=200 \
      --train_freq=20 \
      --lr=$LR \
      --num_layers=4 \
      --hidden_size=64 \
      --input_size=64 \
      --weight_decay=0.001 \
      --train_portion=0.65 \
      --hypotheses_path=$DATA/$accent/manifests/quartznet_outputs/seed_plus_dev_out.txt \
      --lr_decay=warmup \
      --seed=$run \
      --output_dir=$PRETRAINED_CKPTS/error_models/$accent/run_"$run"/recent \
      --best_dir=$PRETRAINED_CKPTS/error_models/$accent/run_"$run"/best \
      --pretrained_ckpt=$PRETRAINED_CKPTS/error_models/librispeech/seed_"$run"/best/ErrorClassifierPhoneBiLSTM_V2.pt
    echo
  done
done
