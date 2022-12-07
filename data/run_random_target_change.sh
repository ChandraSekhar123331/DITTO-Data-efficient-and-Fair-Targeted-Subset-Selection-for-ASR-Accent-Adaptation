homepath=$(pwd)
finetunepath=$(cd ../models/quartznet_asr; pwd)
budget=100
feature="39"
accents=$1
for target in 5 20 50 
do 
	echo
	echo "---------------------------------------------------------"
	echo "$target" "$budget" "$accents" "$feature"
	echo
	python TSS_random.py --target "$target" --budget "$budget" --accent "$accents" --feature_type "$feature"
	#cd "$finetunepath"
	#. scripts/asr_finetune_random.sh
	#. scripts/asr_test_random.sh
	#cd "$homepath"
done
