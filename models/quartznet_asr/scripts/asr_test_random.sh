DATA=$(cd ../../data; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
accent=$1
budget=200
target=20
for run in 1 2 3
do 
	for size in "$budget"
	do
		echo $accent $run $size
		echo 
		echo
		model_dir=$PRETRAINED_CKPTS/quartznet/finetuned/$accent/all/budget_"$size"/target_"$target"/random/run_"$run"/
		python3 -u inference.py \
		--batch_size=64 \
		--output_file=$model_dir/test_out.txt \
		--wav_dir=$DATA/indicTTS_audio/indicTTS/$accent/english/wav \
		--val_manifest=$DATA/$accent/manifests/test.json \
		--model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
		--ckpt=$model_dir/best/Jasper.pt \
		> $model_dir/test_infer_log.txt
		output_dir=$DATA/$accent/manifests/TSS_output/all/budget_"$budget"/target_"$target"/random/run_"$run"/output/
		cp $model_dir/test_infer_log.txt $output_dir
	done
done
# rm -r $PRETRAINED_CKPTS/quartznet/finetuned/mcv-accent_"$accent"/all/budget_$budget/target_"$target"