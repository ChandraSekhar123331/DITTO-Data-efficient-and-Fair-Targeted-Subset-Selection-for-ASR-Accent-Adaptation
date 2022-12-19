
pwd

WAV_DATA=$(cd ../../mozilla/cv-corpus-7.0-2021-07-21/en/wav; pwd)
DATA=$(cd ../../mz-isca/expts/; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd)
accent=$1
# declare -a accents=('african' 'australia' 'bermuda' 'canada' 'england' 'hongkong' 'indian' 'ireland' 'malaysia' 'philippines' 'scotland' 'southatlandtic' 'us' 'wales')
# declare -a accents=('assamese_female_english')
# for accent in "${accents[@]}"
# do
mkdir -p $DATA/$accent/all/quartznet_outputs
echo $accent
python3 -u inference.py \
--batch_size=64 \
--output_file=$DATA/$accent/all/quartznet_outputs/seed_plus_dev_out.txt \
--wav_dir=$WAV_DATA \
--val_manifest=$DATA/$accent/seed_plus_dev.json \
--model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
--ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
> $DATA/$accent/all/quartznet_outputs/seed_plus_dev_infer_log.txt
# done
