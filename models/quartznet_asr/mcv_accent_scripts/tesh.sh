DATA=$(cd ../../mz-expts/; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
accent=$1
echo $accent 
echo 
echo
model_dir=$PRETRAINED_CKPTS/quartznet/finetuned/mcv_pre/$accent

mkdir -p $model_dir

python3 -u inference.py \
--batch_size=64 \
--output_file=$model_dir/test_out.txt \
--wav_dir="" \
--val_manifest=$DATA/$accent/manifests/test.json \
--model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
--ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt 
> $model_dir/test_infer_log.txt
output_dir=$DATA/$accent/manifests/
mv $model_dir/test_infer_log.txt $output_dir/pretrained_test_infer_log.txt

