homepath=$(pwd)
finetunepath=$(cd ../../models/quartznet_asr; pwd)
target=10
feature="39"
accents=$1
for budget in $2 
#for budget in 200 300 400 800
do 
	echo
	echo "---------------------------------------------------------"
	echo "$target" "$budget" "$accents" "$feature"
	echo
#	python TSS_random.py --target "$target" --budget "$budget" --accent "$accents" --feature_type "$feature"
#	for file in $accents/manifests/TSS_output/all/budget_"$budget"/target_"$target"/random/run_*/t*/*;
#	do
#		cp $file ${file/train./dristi_train.}
#		sed -i 's/\/home\/mayank\/MTP\/begin_again/\/mnt\/data\/aman\/mayank\/MTP\/mount_points/g' ${file/train./dristi_train.}
#	done
	cd "$finetunepath"
	. l2_accent_scripts/dristi_asr_finetune_random.sh
	. l2_accent_scripts/dristi_asr_test_random.sh
	cd "$homepath"
done
