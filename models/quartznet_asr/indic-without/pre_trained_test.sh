declare -a accents=('hindi_male_english' 'tamil_male_english' 'rajasthani_male_english' 'malayalam_male_english' 'kannada_male_english' 'assamese_female_english' 'manipuri_female_english' 'gujarati_female_english')
DATA=$(cd ../../indic/speaker-without; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
for accent in "${accents[@]}"
do
	model_dir=$PRETRAINED_CKPTS/quartznet/finetuned/$accent/pre_trained
	mkdir -p $model_dir
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
done
