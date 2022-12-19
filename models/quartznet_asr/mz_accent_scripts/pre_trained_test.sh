declare -a accents=('african' 'indian' 'hongkong' 'philippines' 'england' 'scotland' 'ireland' 'australia' 'canada' 'us' 'bermuda' 'southatlandtic' 'wales' 'malaysia')
DATA=$(cd ../../mz-isca/expts; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
for accent in "${accents[@]}"
do 
	model_dir=$PRETRAINED_CKPTS/quartznet/finetuned/mz-accent_"$accent"/
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
    output_dir=$DATA/$accent/manifests/TSS_output/
    cp $model_dir/test_infer_log.txt $output_dir
done

