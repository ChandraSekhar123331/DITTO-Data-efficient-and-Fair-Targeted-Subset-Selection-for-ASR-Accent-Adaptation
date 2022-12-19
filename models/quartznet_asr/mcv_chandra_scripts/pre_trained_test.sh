DATA=$(cd ../../chandra/MCV/data/; pwd)
WAV_DATA=$(cd ../../mozilla/; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 

accent=$1
rel_file_dir=$2


model_dir=$PRETRAINED_CKPTS/quartznet/pretrained/MCV_chandra/"$rel_file_dir"
mkdir -p $model_dir
echo 
echo $accent
echo 
echo
python3 -u inference.py \
--batch_size=64 \
--output_file=$model_dir/test_out.txt \
--wav_dir="$WAV_DATA"/cv-corpus-7.0-2021-07-21/en/wav/ \
--val_manifest=$DATA/$accent/test.json \
--model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
--ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
> $model_dir/test_infer_log.txt

cp $model_dir/test_infer_log.txt $DATA/$rel_file_dir


