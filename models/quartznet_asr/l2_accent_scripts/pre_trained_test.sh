declare -a accents=('hindi' 'arabic' 'spanish' 'vietnamese' 'korean' 'chinese')
DATA=$(cd ../../CMU_expts/accent-with; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
for accent in "${accents[@]}"
do 
	model_dir=$PRETRAINED_CKPTS/quartznet/finetuned/$accent
	mkdir -p $model_dir
	echo 
	echo $accent
	echo 
	echo
	python3 -u inference.py \
	--batch_size=64 \
	--output_file=$model_dir/test_out.txt \
	--wav_dir="" \
	--val_manifest=$DATA/$accent/manifests/test.json \
	--model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
	--ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
	> $model_dir/test_infer_log.txt
done

