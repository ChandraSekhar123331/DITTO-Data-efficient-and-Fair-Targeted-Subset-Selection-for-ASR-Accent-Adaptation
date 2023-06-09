DATA=$(cd ../../mz-isca/expts; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
declare -a accents=('hindi_male_english' 'tamil_male_english' 'rajasthani_male_english' 'malayalam_male_english' 'kannada_male_english' 'assamese_female_english' 'manipuri_female_english' 'gujarati_female_english')

# declare -a accents=('hindi_male_english')
for accent in "${accents[@]}"
do
  mkdir -p $DATA/$accent/manifests/quartznet_outputs
  echo $accent
  python3 -u inference.py \
  --batch_size=64 \
  --output_file=$DATA/$accent/manifests/quartznet_outputs/all.txt \
  --wav_dir=$DATA/indicTTS_audio/$accent/english/wav \
  --val_manifest=$DATA/$accent/manifests/all.json \
  --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
  --ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
  > $DATA/$accent/manifests/quartznet_outputs/all.txt
done