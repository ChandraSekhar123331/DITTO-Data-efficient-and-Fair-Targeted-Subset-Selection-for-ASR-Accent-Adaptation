WAV_DATA=$(cd ../../mozilla/cv-corpus-7.0-2021-07-21/en/wav; pwd)
DATA=$(cd ../../mz-isca/expts/; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
accent=$1
rel_file_dir=$2
# IN=$rel_file_dir
# arrIN=(${IN//data/ })
# expt_details=${arrIN[1]}
# echo $accent $file_dir

model_dir=$PRETRAINED_CKPTS/quartznet/finetuned/"$rel_file_dir"
python3 -u inference.py \
	--batch_size=64 \
	--output_file=$model_dir/test_out.txt \
	--wav_dir=$WAV_DATA \
	--val_manifest=$DATA/$accent/test.json \
	--model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
	--ckpt=$model_dir/best/Jasper.pt \
	> $model_dir/test_infer_log.txt
	cp $model_dir/test_infer_log.txt $DATA/$rel_file_dir
echo $rel_file_dir/train.json completed
