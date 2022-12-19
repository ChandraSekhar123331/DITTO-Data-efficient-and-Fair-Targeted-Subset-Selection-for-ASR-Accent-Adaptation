DATA=$(cd ../../mz-isca/expts; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
# declare -a accents=('indian' 'hongkong' 'ireland' 'malaysia')
declare -a accents=('malaysia')

for accent in "${accents[@]}"
do
  mkdir -p $DATA/$accent/manifests/quartznet_outputs
  echo $accent
  python3 -u inference.py \
  --batch_size=64 \
  --output_file=$DATA/$accent/manifests/quartznet_outputs/all.txt \
  --wav_dir="" \
  --val_manifest=$DATA/$accent/manifests/all.json \
  --model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
  --ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
  > $DATA/$accent/manifests/quartznet_outputs/all.txt
done