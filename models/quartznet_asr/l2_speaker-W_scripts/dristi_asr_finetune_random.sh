DATA=$(cd ../../CMU_expts/speaker_with; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
for run in 1 2 3
do 
	for size in "$budget"
	do
		echo $speaker $seed $size
		echo 
		echo
		model_dir=$PRETRAINED_CKPTS/quartznet/finetuned/speaker_with_"$speaker"/all/budget_"$size"/target_"$target"/random/run_"$run"/model
		mkdir -p $model_dir
		python3 -u finetune.py \
		--batch_size=16 \
		--num_epochs=100 \
		--eval_freq=1 \
		--train_freq=30 \
		--lr=1e-5 \
		--wav_dir="" \
		--train_manifest=$DATA/$speaker/manifests/TSS_output/all/budget_"$size"/target_"$target"/random/run_"$run"/train/dristi_train.json \
		--val_manifest=$DATA/$speaker/manifests/dristi_dev.json \
		--model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
		--output_dir=$model_dir/recent \
		--best_dir=$model_dir/best \
		--early_stop_patience=10 \
		--zero_infinity \
		--save_after_each_epoch \
		--turn_bn_eval \
		--ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
		--lr_decay=warmup \
		--seed=42 \
		--optimizer=novograd \
		> $model_dir/train_log.txt
	done
done
