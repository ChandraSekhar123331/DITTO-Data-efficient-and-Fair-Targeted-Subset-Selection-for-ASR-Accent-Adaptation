accent=$1
DATA=$(cd ../../data; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
model_dir=$PRETRAINED_CKPTS/quartznet/finetuned/$accent
echo 
echo $accent
echo 
echo
python3 -u inference.py \
--batch_size=64 \
--output_file=$model_dir/test_out.txt \
--wav_dir=$DATA/indicTTS_audio/indicTTS/$accent/english/wav \
--val_manifest=$DATA/$accent/manifests/test.json \
--model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
--ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
> $model_dir/test_infer_log.txt
