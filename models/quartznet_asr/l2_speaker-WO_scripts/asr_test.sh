DATA=$(cd ../../CMU_expts/speaker-without; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
for run in 1 2 3
do 
	for size in "$budget"
	do
		echo $speaker $run $size
		echo 
		echo
		model_dir=$PRETRAINED_CKPTS/quartznet/finetuned/speaker-without_"$speaker"/TSS_output/all/budget_"$size"/target_"$target"/"$fxn"/eta_"$eta"/"$similarity"/"$feature"/run_"$run"
		python3 -u inference.py \
		--batch_size=64 \
		--output_file=$model_dir/test_out.txt \
		--wav_dir="" \
		--val_manifest=$DATA/$speaker/manifests/test.json \
		--model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
		--ckpt=$model_dir/best/Jasper.pt \
		> $model_dir/test_infer_log.txt
		output_dir=$DATA/$speaker/manifests/TSS_output/all/budget_"$budget"/target_"$target"/"$fxn"/eta_"$eta"/"$similarity"/"$feature"/run_"$run"/output/
		cp $model_dir/test_infer_log.txt $output_dir
	done
done
# rm -r $PRETRAINED_CKPTS/quartznet/finetuned/speaker-without_"$speaker"/TSS_output/all/budget_"$size"/target_"$target"/"$fxn"/eta_"$eta"/"$similarity"/"$feature"/run_"$run"/model

