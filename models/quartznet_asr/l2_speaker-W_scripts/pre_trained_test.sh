#declare -a speakers=('ABA' 'ASI' 'BWC' 'EBVS' 'ERMS' 'HJK' 'HKK' 'HQTV' 'LXC' 'MBMPS' 'NCC' 'NJS' 'PNV' 'RRBI' 'SKA' 'SVBI' 'THV' 'TLV' 'TNI' 'TXHC' 'YBAA' 'YDCK' 'YKWK' 'ZHAA')
declare -a speakers=('ABA' 'ASI' 'BWC' 'HJK' 'NJS' 'PNV')
DATA=$(cd ../../CMU_expts/speaker_with; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
for speaker in "${speakers[@]}"
do
	model_dir=$PRETRAINED_CKPTS/quartznet/finetuned/$speaker/pre_trained
	mkdir -p $model_dir 
	echo 
	echo $speaker
	echo 
	ls
	echo $model_dir
	echo
	python3 -u inference.py \
	--batch_size=64 \
	--output_file=$model_dir/test_out.txt \
	--wav_dir="" \
	--val_manifest=$DATA/$speaker/manifests/test.json \
	--model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
	--ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
	> $model_dir/test_infer_log.txt
done
