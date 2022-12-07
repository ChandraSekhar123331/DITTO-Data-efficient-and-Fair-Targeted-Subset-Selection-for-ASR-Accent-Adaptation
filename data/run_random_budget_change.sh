homepath=$(pwd)
finetunepath=$(cd ../models/quartznet_asr; pwd)
target=10
feature="39"
accents=$1
#for budget in 200 300 400 800
for budget in $2
do 
	echo
	echo "---------------------------------------------------------"
	echo "$target" "$budget" "$accents" "$feature"
	echo
	python TSS_random.py --target "$target" --budget "$budget" --accent "$accents" --feature_type "$feature"
	cd "$finetunepath"
	. scripts/asr_finetune_random.sh
	. scripts/asr_test_random.sh
	cd "$homepath"
done
