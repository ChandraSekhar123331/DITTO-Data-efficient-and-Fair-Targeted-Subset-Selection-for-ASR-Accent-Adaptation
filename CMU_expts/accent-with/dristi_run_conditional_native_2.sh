homepath=$(pwd)
finetunepath=$(cd ../../models/quartznet_asr; pwd)
target=10
budget=100
similarity="euclidean"
eta=1.0
feature="39"
accents=$1
declare -a fxns=('FLMI' 'LogDMI')
for fxn in "${fxns[@]}"
do
	echo
	echo "---------------------------------------------------------"
	echo "$fxn"
	echo "$target" "$budget" "$similarity" "$eta" "$accents" "$fxn" "$feature"
	echo
#	python TSS_conditional_native.py --target "$target" --budget "$budget" --similarity "$similarity" --eta "$eta" --accent "$accents" --fxn "$fxn" --feature_type "$feature"
#	for file in $accents/manifests/TSS_output/all/budget_"$budget"/target_"$target"/conditional_"$fxn"_native/eta_"$eta"/"$similarity"/"$feature"/run_*/t*/*;
#	do
#		cp $file ${file/train./dristi_train.}
#		sed -i 's/\/home\/mayank\/MTP\/begin_again/\/mnt\/data\/aman\/mayank\/MTP\/mount_points/g' ${file/train./dristi_train.}
#	done
	cd "$finetunepath"
	. l2_accent_scripts/dristi_asr_finetune_conditional_native.sh
	. l2_accent_scripts/dristi_asr_test_conditional_native.sh
	cd "$homepath"
done
