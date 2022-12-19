# declare -a accents=('hindi_male_english' 'tamil_male_english' 'rajasthani_male_english' 'malayalam_male_english' 'kannada_male_english' 'assamese_female_english' 'manipuri_female_english' 'gujarati_female_english')
# DATA=$(cd ../../data; pwd)
# PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
# for accent in "${accents[@]}"
# do 
# 	output_dir=$DATA/$accent/TSS_output/
# 	mkdir -p $
# 	echo $accent
# 	echo
# 	python3 -u inference.py \
# 	--batch_size=20 \
# 	--output_file=$output_dir/pre_test_out.txt \
# 	--wav_dir="" \
# 	--val_manifest=$DATA/$accent/test.json \
# 	--model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
# 	--ckpt=$PRETRAINED_CKPTS/quartznet/librispeech/quartznet.pt \
# 	> $output_dir/pre_test_infer_log.txt
# done



DATA=$(cd ../../mz-isca/expts/; pwd)
PRETRAINED_CKPTS=$(cd ../pretrained_checkpoints; pwd) 
accent=$1
output_dir="$DATA/$2"

echo "----------------Doing testing with PRETRAINED model.-----------------"
echo "----------------Accent: = $accent----------------"
echo "----------------Output_dir is $output_dir-----------------"

model_dir=$PRETRAINED_CKPTS/quartznet/librispeech

python3 -u inference.py \
	--batch_size=64 \
	--output_file=$output_dir/pretrained_test_out.txt \
	--wav_dir=$DATA/indicTTS_audio/indicTTS/$accent/english/wav \
	--val_manifest=$output_dir/train.json \
	--model_toml=$PRETRAINED_CKPTS/quartznet/quartznet15x5.toml \
	--ckpt=$model_dir/quartznet.pt \
	> $output_dir/pretrained_test_infer_log.txt
echo $output_dir/train.json completed
