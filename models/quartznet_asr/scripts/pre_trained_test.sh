declare -a accents=('hindi_male_english' 'tamil_male_english' 'rajasthani_male_english' 'malayalam_male_english' 'kannada_male_english' 'assamese_female_english' 'manipuri_female_english' 'gujarati_female_english')
DATA=$(cd ../../data; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
for accent in "${accents[@]}"
do 
	output_dir=$DATA/$accent/TSS_output/
	mkdir -p $
	echo $accent
	echo
	python3 -u inference.py \
	--batch_size=20 \
	--output_file=$output_dir/pre_test_out.txt \
	--wav_dir="" \
	--val_manifest=$DATA/$accent/test.json \
	--model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
	--ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
	> $output_dir/pre_test_infer_log.txt
done

